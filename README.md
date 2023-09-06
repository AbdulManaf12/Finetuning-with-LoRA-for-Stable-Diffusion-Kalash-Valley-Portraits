# Finetuning-with-LoRA-for-Stable-Diffusion-Kalash-Valley-Portraits
This repository contains the implementation for fine-tuning a pretrained Stable Diffusion model using LORA (Low Rank Adoption). The goal of this project is to generate high-quality images of people from the Kalash Valley using deep learning techniques.

## Task Overview
* Objective: To fine-tune a pretrained Stable Diffusion model with LORA for generating Kalash Valley portraits. ✅
* Dataset: Gather and process a minimum of 20 images of people from the Kalash Valley. ✅
* Training Method: Utilize distributed training using PyTorch XLA for efficient distributed training. ❌

## Dataset
Our dataset was curated by randomly selecting images from [Pinterest](https://www.pinterest.com/search/pins/?q=kalash%20valley&rs=typed). These images have been manually reviewed and verified for quality. You can access the dataset in the ['KalashValleyImages'](https://github.com/AbdulManaf12/Finetuning-with-LoRA-for-Stable-Diffusion-Kalash-Valley-Portraits/tree/main/KalashValleyImages) folder, you can also check the following image:
 ![dataset big view](https://github.com/AbdulManaf12/Finetuning-with-LoRA-for-Stable-Diffusion-Kalash-Valley-Portraits/blob/main/Input.png)

## Training

For the training phase, we leveraged a publicly available repository on GitHub and fine-tuned it with our custom dataset. The repository we utilized is called [Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning](https://github.com/cloneofsimo/lora), while our choice for the pretrained model came from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5), named 'runwayml/stable-diffusion-v1-5'.

To perform the training, execute the following command:

```console
$ python lora/training_scripts/train_lora_dreambooth.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --instance_data_dir=/path/to/training/images/folder \
  --output_dir=/path/to/output/directory \
  --instance_prompt="Your Prompt Goes Here" \
  --resolution=512 \
  --use_8bit_adam \
  --mixed_precision=fp16 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=0.0003 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --train_text_encoder \
  --lora_rank=16 \
  --learning_rate_text=0.0001 \
  --save_steps 500
```

Please note that due to the limitations of a single GPU in our Colab environment, we did not utilize distributed GPU training. However, if you have access to multiple GPUs and wish to utilize them for training, you can execute the following command:
``` console
$ python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> lora/training_scripts/train_lora_dreambooth.py \
   .... (rest of the same command).
```

Note: Replace <NUM_GPUS> with the number of GPUs you intend to use for distributed training.

I conducted model training for a total of 5000 steps. The architecture used for this training is a variant of the U-Net model, specifically designed for stable diffusion. Below is an overview of the U-Net stable diffusion model:

![image of unet stable diffusion model](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/styles/os_files_xxlarge/public/binxuw/files/stablediffusion_overview.jpg?m=1667438590&itok=n2gM0Xba)
This architecture was selected for its effectiveness in handling the diffusion process, resulting in improved image generation capabilities. 

## Results
Below, you can find the outcomes generated using various prompts:

``` console
PROMPT-1: "girl at kallashV"
```
![girl](https://github.com/AbdulManaf12/Finetuning-with-LoRA-for-Stable-Diffusion-Kalash-Valley-Portraits/blob/main/Output-1.png)

``` console
PROMPT-2: "boy at kallashV"
```
![girl](https://github.com/AbdulManaf12/Finetuning-with-LoRA-for-Stable-Diffusion-Kalash-Valley-Portraits/blob/main/Output-2.png)

``` console
PROMPT-3: "people at kallashV"
```
![girl](https://github.com/AbdulManaf12/Finetuning-with-LoRA-for-Stable-Diffusion-Kalash-Valley-Portraits/blob/main/Output-3.png)

Intriguingly, these results demonstrate the model's capacity to discern patterns from a limited number of instances, underscoring its adaptability and learning capabilities.

## Acknowledgments

- I would like to acknowledge the work of the [LoRA repository](https://github.com/cloneofsimo/lora) by [cloneofsimo](https://github.com/cloneofsimo) for their implementation of Low Rank Adoption (LoRA) that was used in this project. My work builds upon and incorporates their code as a crucial component of the fine-tuning process.
