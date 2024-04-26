from typing import List, Optional, Tuple, overload, Union

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn

from vlmrm.contrib.open_clip.transform import image_transform
from vlmrm.trainer.config import CLIPRewardConfig
from PIL import Image
import numpy as np
from torch.nn.functional import normalize
from transformers import AutoModel, AutoProcessor, BatchFeature


class CLIPEmbed(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        if isinstance(clip_model.visual.image_size, int):
            image_size = clip_model.visual.image_size
        else:
            image_size = clip_model.visual.image_size[0]
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
            # x = x.cpu()
            # print(x[0].shape)
            # image = Image.fromarray(x[0].permute(1, 2, 0).numpy(), 'RGB')
            # print(image)
            # print(type(image))
            # image.save('output_image.png', 'PNG')
            # import pdb; pdb.set_trace()

        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.transform(x)
            x = self.clip_model.encode_image(x, normalize=True)
        return x


class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,
        reward_func: str,
        sparse: bool,
        threshold: float,
        alpha: float,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        self.reward_func = reward_func
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold
        target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
        baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)


        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)
    
    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the output of CLIPEmbed (image embedding)
        x = x / torch.norm(x, dim=-1, keepdim=True) 
        if self.reward_func == "contrastive":
            sim_s_g = nn.functional.cosine_similarity(x, self.target)
            sim_s_b = nn.functional.cosine_similarity(x, self.baseline)
            P_s_g = torch.exp(sim_s_g) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
            P_s_b = torch.exp(sim_s_b) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
            P_s = (torch.exp(sim_s_g) + torch.exp(sim_s_b)) / (torch.exp(sim_s_g) + torch.exp(sim_s_b) + 1)
            P_g = torch.tensor(0.5)
            P_b = torch.tensor(0.5)
            NPMI_s_g = (torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g))
            NPMI_s_b = (torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b))
            y = NPMI_s_g - self.alpha * NPMI_s_b
        if self.reward_func == "goal_baseline_reg":
            y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        elif self.reward_func == "cosine":
            y = nn.functional.cosine_similarity(x, self.target)
        elif self.reward_func == "l2":
            y = 1 / nn.functional.pairwise_distance(x, self.target)
       
        if not self.sparse:
            return y
        else:
            return torch.where(y > self.threshold, 1.0, 0.0)

    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return open_clip.tokenize(x)

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.embed_module.clip_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.embed_module.forward(x)
        

class SigLipReward(nn.Module):
    def __init__(
        self,
        *,
        model: AutoModel,
        processor: AutoProcessor,
        reward_func: str,
        sparse: bool,
        threshold: float,
        alpha: float,
        multi_prompt: bool,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        self.processor = processor
        self.reward_func = reward_func
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold
        self.multi_prompt = multi_prompt
        self.target_prompts = target_prompts
        self.baseline_prompts = baseline_prompts
        if self.reward_func == "goal_baseline_reg" or not self.multi_prompt:
            target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
            baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)

        else:
            target = self.embed_prompts(target_prompts)
            baseline = self.embed_prompts(baseline_prompts)
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)


        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)
    
    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.processor.image_processor(x)
        # with torch.no_grad():
        #     # print(x)
        #     print(type(x), "~~~~~~~~~~~~~~~~~~~~~~~~~")
        #     x = self.embed_module.get_image_features(**x) # equivalent of a CLIPEmbed (image embedding)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        if self.reward_func == "contrastive":
            if self.multi_prompt:

                sim_s_g = torch.stack([nn.functional.cosine_similarity(x, target_embedding) for target_embedding in self.target])
                sim_s_b = torch.stack([nn.functional.cosine_similarity(x, baseline_embedding) for baseline_embedding in self.baseline])
            else:
                sim_s_g = nn.functional.cosine_similarity(x, self.target)
                sim_s_b = nn.functional.cosine_similarity(x, self.baseline)


            if self.multi_prompt:
                # Compute probabilities for each target and baseline pair
                P_s_g_list = []
                P_s_b_list = []
                P_s_list = []
                for sim_g, sim_b in zip(sim_s_g, sim_s_b):
                    P_s_g = torch.exp(sim_g) / (torch.exp(sim_g) + torch.exp(sim_b))
                    P_s_b = torch.exp(sim_b) / (torch.exp(sim_g) + torch.exp(sim_b))
                    P_s = (torch.exp(sim_g) + torch.exp(sim_b)) / (torch.exp(sim_g) + torch.exp(sim_b) + 1)
                    P_s_g_list.append(P_s_g)
                    P_s_b_list.append(P_s_b)
                    P_s_list.append(P_s)
                
                # Stack the probabilities along a new dimension
                P_s_g = torch.stack(P_s_g_list)
                P_s_b = torch.stack(P_s_b_list)
                P_s = torch.stack(P_s_list)
            else:
                # Compute probabilities for the single target and baseline
                P_s_g = torch.exp(sim_s_g) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
                P_s_b = torch.exp(sim_s_b) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
                P_s = (torch.exp(sim_s_g) + torch.exp(sim_s_b)) / (torch.exp(sim_s_g) + torch.exp(sim_s_b) + 1)

            P_g = torch.tensor(0.5)
            P_b = torch.tensor(0.5)

            # Compute NPMI using log probabilities directly for stability
            if self.multi_prompt:
                NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
                NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)
                
                # # Take the mean NPMI across the multiple prompts
                NPMI_s_g = NPMI_s_g.mean(dim=0)
                NPMI_s_b = NPMI_s_b.mean(dim=0)
            else:
                NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
                NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)

            y = NPMI_s_g - self.alpha * NPMI_s_b
            # sim_s_g = nn.functional.cosine_similarity(x, self.target)
            # sim_s_b = nn.functional.cosine_similarity(x, self.baseline)
            # # Estimate probabilities
            # P_s_g = torch.exp(sim_s_g) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
            # P_s_b = torch.exp(sim_s_b) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
            # P_s = (torch.exp(sim_s_g) + torch.exp(sim_s_b)) / (torch.exp(sim_s_g) + torch.exp(sim_s_b) + 1)
            # P_g = torch.tensor(0.5)
            # P_b = torch.tensor(0.5)
            # # print(sim_s_b.shape, sim_s_g.shape)
            # # Compute NPMI
            # NPMI_s_g = (torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g))
            # NPMI_s_b = (torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b))
            # # Compute regularized reward
            # y = NPMI_s_g - self.alpha * NPMI_s_b
        if self.reward_func == "goal_baseline_reg":
            y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        elif self.reward_func == "cosine":
            if self.multi_prompt:
                cosines = torch.stack([torch.nn.functional.cosine_similarity(x, t) for t in self.target])
                y = cosines.mean()
            else:
                y = nn.functional.cosine_similarity(x, self.target)
        elif self.reward_func == "l2":
            y = 1 / nn.functional.pairwise_distance(x, self.target)
        
        if not self.sparse:
            return y
        else:
            return torch.where(y > self.threshold, 1.0, 0.0)

    def tokenize_prompts(self, x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return self.processor.tokenizer(x, padding="max_length", return_tensors="pt")["input_ids"]

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        inputs = self.processor.tokenizer(x, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            x = self.embed_module.get_text_features(**inputs)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x, rank=None):
        x = self.processor.image_processor(x)
        y = BatchFeature(data=dict(**x), tensor_type="pt")
        with torch.no_grad():
            if rank:
                x = self.embed_module.get_image_features(y["pixel_values"].cuda(rank))
            else:
                x = self.embed_module.get_image_features(y["pixel_values"].cuda())
        return x

class FLAVAReward(nn.Module):
    def __init__(
        self,
        *,
        model: AutoModel,
        processor: AutoProcessor,
        reward_func: str,
        sparse: bool,
        threshold: float,
        alpha: float,
        multi_prompt: bool,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        self.processor = processor
        self.reward_func = reward_func
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold
        self.multi_prompt = multi_prompt
        self.target_prompts = target_prompts
        self.baseline_prompts = baseline_prompts
        if self.multi_prompt:
            target = self.embed_prompts(target_prompts)
            baseline = self.embed_prompts(baseline_prompts)
        else:
            target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
            baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)


        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)
    
    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.processor.image_processor(x)
        # with torch.no_grad():
        #     # print(x)
        #     print(type(x), "~~~~~~~~~~~~~~~~~~~~~~~~~")
        #     x = self.embed_module.get_image_features(**x) # equivalent of a CLIPEmbed (image embedding)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        if self.reward_func == "contrastive":
            if self.multi_prompt:
                sim_s_g = torch.stack([nn.functional.cosine_similarity(x, target_embedding) for target_embedding in self.target])
                sim_s_b = torch.stack([nn.functional.cosine_similarity(x, baseline_embedding) for baseline_embedding in self.baseline])
            else:
                sim_s_g = nn.functional.cosine_similarity(x, self.target)
                sim_s_b = nn.functional.cosine_similarity(x, self.baseline)


            if self.multi_prompt:
                # Compute probabilities for each target and baseline pair
                P_s_g_list = []
                P_s_b_list = []
                P_s_list = []
                for sim_g, sim_b in zip(sim_s_g, sim_s_b):
                    P_s_g = torch.exp(sim_g) / (torch.exp(sim_g) + torch.exp(sim_b))
                    P_s_b = torch.exp(sim_b) / (torch.exp(sim_g) + torch.exp(sim_b))
                    P_s = (torch.exp(sim_g) + torch.exp(sim_b)) / (torch.exp(sim_g) + torch.exp(sim_b) + 1)
                    P_s_g_list.append(P_s_g)
                    P_s_b_list.append(P_s_b)
                    P_s_list.append(P_s)
                
                # Stack the probabilities along a new dimension
                P_s_g = torch.stack(P_s_g_list)
                P_s_b = torch.stack(P_s_b_list)
                P_s = torch.stack(P_s_list)
            else:
                # Compute probabilities for the single target and baseline
                P_s_g = torch.exp(sim_s_g) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
                P_s_b = torch.exp(sim_s_b) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
                P_s = (torch.exp(sim_s_g) + torch.exp(sim_s_b)) / (torch.exp(sim_s_g) + torch.exp(sim_s_b) + 1)

            P_g = torch.tensor(0.5)
            P_b = torch.tensor(0.5)

            # Compute NPMI using log probabilities directly for stability
            if self.multi_prompt:
                NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
                NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)
                
                # # Take the mean NPMI across the multiple prompts
                NPMI_s_g = NPMI_s_g.mean(dim=0)
                NPMI_s_b = NPMI_s_b.mean(dim=0)
            else:
                NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
                NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)

            y = NPMI_s_g - self.alpha * NPMI_s_b
            # P_s_g = torch.exp(sim_s_g) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
            # P_s_b = torch.exp(sim_s_b) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
            # P_s = (torch.exp(sim_s_g) + torch.exp(sim_s_b)) / (torch.exp(sim_s_g) + torch.exp(sim_s_b) + 1)
            # P_g = torch.tensor(0.5)
            # P_b = torch.tensor(0.5)
            # # Compute NPMI
            # NPMI_s_g = (torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g))
            # NPMI_s_b = (torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b))
            # # Compute regularized reward
            # y = NPMI_s_g - self.alpha * NPMI_s_b
        if self.reward_func == "goal_baseline_reg":
            y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        elif self.reward_func == "cosine":
            y = nn.functional.cosine_similarity(x, self.target)
        elif self.reward_func == "l2":
            y = 1 / nn.functional.pairwise_distance(x, self.target)
        
        if not self.sparse:
            return y
        else:
            return torch.where(y > self.threshold, 1.0, 0.0)

    def tokenize_prompts(self, x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return self.processor.tokenizer(x, return_tensors="pt", padding=True, max_length=77)["input_ids"]
        # return self.processor.tokenizer(x, padding="max_length", return_tensors="pt")["input_ids"]

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        inputs = self.processor.tokenizer(x, return_tensors="pt", padding=True, max_length=77)["input_ids"]
        with torch.no_grad():
            x = self.embed_module.get_text_features(inputs)[:,0]
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x, rank=None):
        x = self.processor.feature_extractor(x)
        # y = BatchFeature(data=dict(**x), tensor_type="pt")
        with torch.no_grad():
            if rank:
                x = self.embed_module.get_image_features(torch.tensor(np.array(x["pixel_values"])).cuda(rank))[:,0]
            else:
                x = self.embed_module.get_image_features(torch.tensor(np.array(x["pixel_values"])).cuda())[:,0]
        return x
    
# class BLIPReward(nn.Module):
#     def __init__(
#         self,
#         *,
#         model: AutoModel,
#         processor: AutoProcessor,
#         reward_func: str,
#         sparse: bool,
#         threshold: float,
#         alpha: float,
#         multi_prompt: bool,
#         target_prompts: torch.Tensor,
#         baseline_prompts: torch.Tensor,
#     ) -> None:
#         """CLIP Reward function that modifies the CLIP vector space by
#         projecting all vectors onto the line spanned by the prompt and
#         a baseline prompt. The alpha parameter controls the degree of
#         projection. A value of 0.0 means that the reward function is
#         equivalent to the CLIP reward function. A value of 1.0 means
#         that the vector space is completely projected onto the line
#         and becomes a 1D space. Any value in between is a linear
#         interpolation between the two.

#         Args:
#             model (str): CLIP model.
#             device (str): Device to use.
#             alpha (float, optional): Coeefficient of projection.
#             target_prompts (torch.Tensor): Tokenized prompts describing
#                 the target state.
#             baseline_prompts (torch.Tensor): Tokenized prompts describing
#                 the baseline state.
#         """
#         super().__init__()
#         self.embed_module = model
#         self.processor = processor
#         self.reward_func = reward_func
#         self.sparse = sparse
#         if self.sparse:
#             self.threshold = threshold
#         self.target_prompts = target_prompts
#         self.baseline_prompts = baseline_prompts
#         target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
#         baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
#         direction = target - baseline
#         # Register them as buffers so they are automatically moved around.
#         self.register_buffer("target", target)
#         self.register_buffer("baseline", baseline)
#         self.register_buffer("direction", direction)


#         self.alpha = alpha
#         projection = self.compute_projection(alpha)
#         self.register_buffer("projection", projection)
    
#     def compute_projection(self, alpha: float) -> torch.Tensor:
#         projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
#         identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
#         projection = alpha * projection + (1 - alpha) * identity
#         return projection

#     def update_alpha(self, alpha: float) -> None:
#         self.alpha = alpha
#         self.projection = self.compute_projection(alpha)

#     @torch.inference_mode()
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x / torch.norm(x, dim=-1, keepdim=True)
#         if self.reward_func == "contrastive":
#             if self.multi_prompt:
#                 sim_s_g = torch.stack([nn.functional.cosine_similarity(x, target_embedding) for target_embedding in self.target])
#                 sim_s_b = torch.stack([nn.functional.cosine_similarity(x, baseline_embedding) for baseline_embedding in self.baseline])
#             else:
#                 sim_s_g = nn.functional.cosine_similarity(x, self.target)
#                 sim_s_b = nn.functional.cosine_similarity(x, self.baseline)


#             if self.multi_prompt:
#                 # Compute probabilities for each target and baseline pair
#                 P_s_g_list = []
#                 P_s_b_list = []
#                 P_s_list = []
#                 for sim_g, sim_b in zip(sim_s_g, sim_s_b):
#                     P_s_g = torch.exp(sim_g) / (torch.exp(sim_g) + torch.exp(sim_b))
#                     P_s_b = torch.exp(sim_b) / (torch.exp(sim_g) + torch.exp(sim_b))
#                     P_s = (torch.exp(sim_g) + torch.exp(sim_b)) / (torch.exp(sim_g) + torch.exp(sim_b) + 1)
#                     P_s_g_list.append(P_s_g)
#                     P_s_b_list.append(P_s_b)
#                     P_s_list.append(P_s)
                
#                 # Stack the probabilities along a new dimension
#                 P_s_g = torch.stack(P_s_g_list)
#                 P_s_b = torch.stack(P_s_b_list)
#                 P_s = torch.stack(P_s_list)
#             else:
#                 # Compute probabilities for the single target and baseline
#                 P_s_g = torch.exp(sim_s_g) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
#                 P_s_b = torch.exp(sim_s_b) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
#                 P_s = (torch.exp(sim_s_g) + torch.exp(sim_s_b)) / (torch.exp(sim_s_g) + torch.exp(sim_s_b) + 1)

#             P_g = torch.tensor(0.5)
#             P_b = torch.tensor(0.5)

#             # Compute NPMI using log probabilities directly for stability
#             if self.multi_prompt:
#                 NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
#                 NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)
                
#                 # # Take the mean NPMI across the multiple prompts
#                 NPMI_s_g = NPMI_s_g.mean(dim=0)
#                 NPMI_s_b = NPMI_s_b.mean(dim=0)
#             else:
#                 NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
#                 NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)

#             y = NPMI_s_g - self.alpha * NPMI_s_b
#         if self.reward_func == "goal_baseline_reg":
#             y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
#         elif self.reward_func == "cosine":
#             y = nn.functional.cosine_similarity(x, self.target)
#         elif self.reward_func == "l2":
#             y = 1 / nn.functional.pairwise_distance(x, self.target)
        
#         if not self.sparse:
#             return y
#         else:
#             return torch.where(y > self.threshold, 1.0, 0.0)

#     def tokenize_prompts(self, x: List[str]) -> torch.Tensor:
#         """Tokenize a list of prompts."""
#         return self.processor(text=x, return_tensors="pt", padding=True)["input_ids"]
#         # return self.processor.tokenizer(x, padding="max_length", return_tensors="pt")["input_ids"]

#     def embed_prompts(self, x) -> torch.Tensor:
#         """Embed a list of prompts."""
#         inputs = self.processor(text=x, return_tensors="pt", padding=True)["input_ids"]
#         with torch.no_grad():
#             x = self.embed_module.get_text_features(inputs)
#         x = x / x.norm(dim=-1, keepdim=True)
#         return x

#     def embed_images(self, x, rank=None):
#         x = self.processor(images=x, return_type="pt")
#         with torch.no_grad():
#             if rank:
#                 x = self.embed_module.get_image_features(x["pixel_values"].cuda(rank))
#             else:
#                 x = self.embed_module.get_image_features(x["pixel_values"].cuda())
#         return x

class BLIPReward(nn.Module):
    def __init__(
        self,
        *,
        model: AutoModel,
        processor: AutoProcessor,
        reward_func: str,
        sparse: bool,
        threshold: float,
        alpha: float,
        multi_prompt: bool,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        self.processor = processor
        self.reward_func = reward_func
        self.sparse = sparse
        if self.sparse:
            self.threshold = threshold
        self.multi_prompt = multi_prompt
        self.target_prompts = target_prompts
        self.baseline_prompts = baseline_prompts
        if self.multi_prompt:
            target = self.embed_prompts(target_prompts)
            baseline = self.embed_prompts(baseline_prompts)
        else:
            target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
            baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)


        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)
    
    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, vision_output) -> torch.Tensor:
        # x = self.processor.image_processor(x)
        # with torch.no_grad():
        #     # print(x)
        #     print(type(x), "~~~~~~~~~~~~~~~~~~~~~~~~~")
        #     x = self.embed_module.get_image_features(**x) # equivalent of a CLIPEmbed (image embedding)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        if self.reward_func == "contrastive":
            if self.multi_prompt:
                sim_s_g = torch.stack([nn.functional.cosine_similarity(x, target_embedding) for target_embedding in self.target])
                sim_s_b = torch.stack([nn.functional.cosine_similarity(x, baseline_embedding) for baseline_embedding in self.baseline])
            else:
                sim_s_g = nn.functional.cosine_similarity(x, self.target)
                sim_s_b = nn.functional.cosine_similarity(x, self.baseline)


            if self.multi_prompt:
                # Compute probabilities for each target and baseline pair
                P_s_g_list = []
                P_s_b_list = []
                P_s_list = []
                for sim_g, sim_b in zip(sim_s_g, sim_s_b):
                    P_s_g = torch.exp(sim_g) / (torch.exp(sim_g) + torch.exp(sim_b))
                    P_s_b = torch.exp(sim_b) / (torch.exp(sim_g) + torch.exp(sim_b))
                    P_s = (torch.exp(sim_g) + torch.exp(sim_b)) / (torch.exp(sim_g) + torch.exp(sim_b) + 1)
                    P_s_g_list.append(P_s_g)
                    P_s_b_list.append(P_s_b)
                    P_s_list.append(P_s)
                
                # Stack the probabilities along a new dimension
                P_s_g = torch.stack(P_s_g_list)
                P_s_b = torch.stack(P_s_b_list)
                P_s = torch.stack(P_s_list)
            else:
                # Compute probabilities for the single target and baseline
                P_s_g = torch.exp(sim_s_g) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
                P_s_b = torch.exp(sim_s_b) / (torch.exp(sim_s_g) + torch.exp(sim_s_b))
                P_s = (torch.exp(sim_s_g) + torch.exp(sim_s_b)) / (torch.exp(sim_s_g) + torch.exp(sim_s_b) + 1)

            P_g = torch.tensor(0.5)
            P_b = torch.tensor(0.5)

            # Compute NPMI using log probabilities directly for stability
            if self.multi_prompt:
                NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
                NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)
                
                # # Take the mean NPMI across the multiple prompts
                NPMI_s_g = NPMI_s_g.mean(dim=0)
                NPMI_s_b = NPMI_s_b.mean(dim=0)
            else:
                NPMI_s_g = torch.log(P_s_g / (P_s * P_g)) / -torch.log(P_s_g)
                NPMI_s_b = torch.log(P_s_b / (P_s * P_b)) / -torch.log(P_s_b)

            y = NPMI_s_g - self.alpha * NPMI_s_b
        if self.reward_func == "goal_baseline_reg":
            y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        elif self.reward_func == "cosine":
            y = nn.functional.cosine_similarity(x, self.target)
        elif self.reward_func == "l2":
            y = 1 / nn.functional.pairwise_distance(x, self.target)
        
        elif self.reward_func == "itm_head":
            input_ids = self.tokenize_prompts(self.target_prompts)
            image_atts = torch.ones(x.size()[:-1], dtype=torch.long)
            itm_embeds = self.embed_module.text_encoder(
                input_ids=input_ids,
                encoder_hidden_states=vision_output,
                encoder_attention_mask=image_atts,
            )[0]
            y = self.embed_module.itm_head(itm_embeds[:, 0, :])[:,0]
            # print(y)
        
        if not self.sparse:
            return y
        else:
            return torch.where(y > self.threshold, 1.0, 0.0)

    def tokenize_prompts(self, x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return self.processor.tokenizer(x, return_tensors="pt")["input_ids"]

        # return self.processor.tokenizer(x, padding="max_length", return_tensors="pt")["input_ids"]

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        inputs = self.processor.tokenizer(x, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            question_embeds = self.embed_module.text_encoder(input_ids=inputs)[0]
            x = normalize(self.embed_module.text_proj(question_embeds[:, 0, :]), dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x, rank=None):
        processed_images = []
        for i in range(x.size(0)):
            processed_image = self.processor.image_processor.preprocess(images=x[i], size=[224, 224], return_tensors="pt")
            processed_images.append(processed_image['pixel_values'])
        x = torch.stack(processed_images).squeeze(1)
        with torch.no_grad():
            if rank:
                vision_outputs = self.embed_module.vision_model(pixel_values=x.cuda(rank))[0]
                x = normalize(self.embed_module.vision_proj(vision_outputs[:, 0, :].cuda(rank)), dim=-1)
            else:
                vision_outputs = self.embed_module.vision_model(pixel_values=x.cuda(rank))[0]
                x = normalize(self.embed_module.vision_proj(vision_outputs[:, 0, :].cuda(rank)), dim=-1)
        return x, vision_outputs
    


def load_reward_model(
    model_name, 
    reward_func, 
    sparse,
    threshold,
    multi_prompt,
    target_prompts, 
    baseline_prompts, 
    alpha, 
    cache_dir: Optional[str] = None
):
    model_name_prefix, pretrained = model_name.split("/")
    print(model_name_prefix, pretrained)

    if "siglip" in pretrained.lower():
        from transformers import AutoProcessor
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        model = SigLipReward(model=model, 
                             processor=processor, 
                             reward_func=reward_func, 
                             sparse=sparse,
                             threshold=threshold,
                             alpha=alpha, 
                             multi_prompt=multi_prompt, 
                             target_prompts=target_prompts, 
                             baseline_prompts=baseline_prompts
                            )
        return model.eval()

    elif "flava" in pretrained.lower():
        from transformers import AutoProcessor, FlavaModel
        model = FlavaModel.from_pretrained("facebook/flava-full")
        processor = AutoProcessor.from_pretrained("facebook/flava-full")
        model = FLAVAReward(model=model, 
                             processor=processor, 
                             reward_func=reward_func, 
                             sparse=sparse,
                             threshold=threshold,
                             alpha=alpha, 
                             multi_prompt=multi_prompt, 
                             target_prompts=target_prompts, 
                             baseline_prompts=baseline_prompts)
        return model.eval()
    
    elif "blip" in pretrained.lower():
        from transformers import AutoProcessor, BlipForImageTextRetrieval
        from torch.nn.functional import normalize

        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        model = BLIPReward(model=model, 
                             processor=processor, 
                             reward_func=reward_func, 
                             sparse=sparse,
                             threshold=threshold,
                             alpha=alpha, 
                             multi_prompt=multi_prompt, 
                             target_prompts=target_prompts, 
                             baseline_prompts=baseline_prompts)
        return model.eval()

    else:
        model = open_clip.create_model(
            model_name=model_name_prefix, pretrained=pretrained,cache_dir=cache_dir
        )

    target_prompts = CLIPReward.tokenize_prompts(target_prompts)
    baseline_prompts = CLIPReward.tokenize_prompts(baseline_prompts)
    model = CLIPEmbed(model)
    model = CLIPReward(
        model=model,
        reward_func=reward_func,
        sparse=sparse,
        threshold=threshold,
        alpha=alpha,
        target_prompts=target_prompts,
        baseline_prompts=baseline_prompts,
    )
    return model.eval()


def load_reward_model_from_config(config: CLIPRewardConfig) -> Union[CLIPReward, SigLipReward]:
    return load_reward_model(
        model_name=config.pretrained_model,
        reward_func=config.reward_func,
        sparse=config.sparse,
        threshold=config.threshold,
        multi_prompt=config.multi_prompt,
        target_prompts=config.target_prompts,
        baseline_prompts=config.baseline_prompts,
        alpha=config.alpha,
        cache_dir=config.cache_dir,
    )


def compute_rewards(
    model: Union[CLIPEmbed, SigLipReward, FLAVAReward, BLIPReward],
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                reward_model=model,
                render_dim=frames_batch.shape[1:],
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()
            rewards[i : i + batch_size] = rewards_batch 
    return rewards
    # assert frames.device == torch.device("cpu")
    # assert batch_size % num_workers == 0
    # n_samples = len(frames)
    # rewards = torch.zeros(n_samples, device=model.target.device)
    # model = model.eval()
    # with torch.no_grad():
    #     for i in range(0, n_samples, batch_size):
    #         frames_batch = frames[i : i + batch_size]
    #         rewards_batch = dist_worker_compute_reward(
    #             rank=0,
    #             reward_model=model,
    #             render_dim=frames_batch.shape[1:],
    #             batch_size=batch_size // num_workers,
    #             num_workers=num_workers,
    #             frames=frames_batch,
    #             worker_frames_tensor=worker_frames_tensor,
    #         )
    #         rewards[i : i + batch_size] = rewards_batch

    # return rewards.cpu()


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None,
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        scatter_list = []

    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)
    with torch.no_grad():

        if type(reward_model) == CLIPReward:
            embeddings = reward_model.embed_module(worker_frames)
            rewards = reward_model(embeddings)
        elif type(reward_model) == BLIPReward:
            embeddings, attention = reward_model.embed_images(worker_frames, rank)
            rewards = reward_model(embeddings, attention)

        else:
            embeddings = reward_model.embed_images(worker_frames, rank)
            rewards = reward_model(embeddings)

    def zero_t():
        return torch.zeros_like(rewards)

    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    rewards = rewards.contiguous()
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        return torch.cat(recv_rewards, dim=0).cuda(rank)

