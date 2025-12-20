QLoRA Fine-Tuning with Unsloth: A Complete Guide
Learn how to fine-tune a 3B parameter model using just 8GB of VRAM with 4-bit quantization and optimized LoRA adapters
Matteo Ferruccio Andreoni
Matteo Ferruccio Andreoni

Follow
9 min read
·
2 days ago
53




Press enter or click to view image in full size

Intro
Fine-tuning large language models has traditionally required massive computational resources, putting state-of-the-art AI capabilities out of reach for many developers and researchers. But what if you could fine-tune a 3-billion parameter model on a consumer GPU with just 8GB of VRAM?

Enter QLoRA (Quantized Low-Rank Adaptation) combined with Unsloth — a powerful duo that makes efficient LLM fine-tuning accessible to everyone. In this comprehensive tutorial, I’ll walk you through every step of fine-tuning Llama 3.2 3B using these cutting-edge techniques.

What Makes QLoRA Revolutionary?
Before diving into the code, let’s understand why QLoRA is a game-changer for the AI community.

The Memory Problem
Traditional fine-tuning updates all model parameters, requiring enormous GPU memory. A 3B parameter model in 16-bit precision needs approximately 6GB just to store weights, plus additional memory for gradients, optimizer states, and activations during training. This quickly balloons to 20–30GB or more.

The QLoRA Solution
QLoRA solves this through three key innovations:

1. 4-bit NormalFloat (NF4) Quantization: Neural network weights typically follow a normal distribution. NF4 is an information-theoretically optimal 4-bit data type designed specifically for this distribution, providing better representation than standard 4-bit integers.

2. Double Quantization: Even the quantization constants are quantized, saving an additional 0.37 bits per parameter on average.

3. Paged Optimizers: Manages memory spikes during training using NVIDIA’s unified memory feature, preventing out-of-memory errors during gradient checkpointing.

The result? You can fine-tune models using 4x less memory than standard LoRA, with zero accuracy degradation.

Enter Unsloth: Speed Meets Efficiency
Unsloth takes QLoRA to the next level with hand-optimized kernels that provide:

2x faster training compared to standard implementations
Up to 70% less memory usage
Zero accuracy loss — full compatibility with HuggingFace ecosystem
Support for the latest models including Llama 3.2, Mistral, Gemma, and more

Prerequisites
To follow this tutorial, you’ll need:

An NVIDIA GPU (You can use T4 instance on Google Colab)
CUDA installed and configured
Python 3.8 or later
Basic understanding of neural networks and PyTorch
My ready-to-use repository :)
GitHub - M1T8E6/Unsloth-FineTuning: LoRA & QLoRA Fine‑tuning on NVIDIA GPU (CUDA) with Unsloth &…
LoRA & QLoRA Fine‑tuning on NVIDIA GPU (CUDA) with Unsloth & TRL - train small‑to‑mid LLMs, with a clean notebook…
github.com

Step 1: Environment Setup
First, let’s install Unsloth and its dependencies. The key libraries include:

pip install --no-deps bitsandbytes accelerate xformers peft trl unsloth
pip install sentencepiece protobuf datasets huggingface_hub
These packages provide:

bitsandbytes: 4-bit quantization support
accelerate: Distributed training utilities
xformers: Memory-efficient attention mechanisms
peft: Parameter-Efficient Fine-Tuning implementations
trl: Supervised Fine-Tuning trainer
After installation, verify CUDA availability:

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
Step 2: Loading the Model with 4-bit Quantization
Here’s where QLoRA magic begins. We load a pre-quantized Llama 3.2 3B model:

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect: bfloat16 for Ampere+, float16 for older GPUs
    load_in_4bit=True,  # Enable 4-bit quantization
)
The -bnb-4bit suffix indicates this model is already quantized using BitsAndBytes NF4 format, saving us from having to quantize it ourselves.

Understanding dtype Auto-Detection

Unsloth automatically selects the optimal precision:

bfloat16 for Ampere and newer architectures (RTX 3090+, A100)
float16 for older GPUs (V100, T4)
Bfloat16 offers better numerical stability and is preferred when available.

Step 3: Configuring LoRA Adapters
Now we add trainable LoRA adapters to our frozen 4-bit model. These small matrices are inserted into specific layers, allowing efficient fine-tuning.

LORA_R = 16  # Rank of LoRA matrices
LORA_ALPHA = 16  # Scaling factor
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,  # 0 is optimized for speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
)
The Power of Parameter Efficiency

Let’s check how many parameters we’re actually training:

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%)")
You’ll typically see something like: 41M trainable out of 3.2B total (1.28%). We’re only updating about 1% of the model’s parameters!

LoRA Hyperparameters Explained

Rank (r): The dimensionality of the low-rank matrices. Higher rank means:

More capacity to learn task-specific patterns
More parameters to train
Higher memory usage
Common values: 8, 16, 32, 64. Start with 16 for most tasks.

Alpha: Scaling factor applied to LoRA weights. Often set equal to or 2x the rank. Controls how much influence the adapters have.

Target Modules: Which layers receive LoRA adapters. Targeting both attention and MLP layers provides maximum flexibility.

Step 4: Dataset Preparation
For this tutorial, we use the FineTome-100k dataset, a curated collection of high-quality instruction-response pairs:

from datasets import load_dataset

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
print(f"Dataset size: {len(dataset):,} examples")
Chat Templates: The Secret Sauce

Each model family uses specific formatting for conversations. Using the wrong template can significantly hurt performance. Unsloth makes this easy:

from unsloth.chat_templates import get_chat_template, standardize_sharegpt

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  # Llama 3.2 uses 3.1 template
)

dataset = standardize_sharegpt(dataset)
This applies the proper <|start_header_id|> and <|end_header_id|> markers that Llama models expect.

Formatting the Dataset

We need to convert conversations into the model’s expected text format:

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
Step 5: Training Configuration
Now we configure the training parameters, optimized for QLoRA:

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    output_dir="./outputs_qlora",
    
    # Batch settings
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    
    # Training duration
    max_steps=60,  # For demo; use num_train_epochs=3 for full training
    
    # Learning
    learning_rate=2e-4,
    warmup_steps=10,
    lr_scheduler_type="linear",
    
    # Precision
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    
    # Optimizer: 8-bit saves memory
    optim="adamw_8bit",
    weight_decay=0.01,
    
    # Logging
    logging_steps=10,
    seed=3407,
)
Understanding Effective Batch Size

With per_device_train_batch_size=2 and gradient_accumulation_steps=4:

Effective Batch Size = 2 × 4 × num_gpus = 8
Gradient accumulation lets us achieve larger effective batch sizes without increasing memory usage — gradients are accumulated over multiple forward passes before updating weights.

The 8-bit Optimizer Trick

Using adamw_8bit quantizes the optimizer states to 8-bit, saving substantial memory. Adam optimizer typically stores two 32-bit states per parameter (momentum and variance), so 8-bit quantization provides 4x memory savings here alone.

Step 6: Initializing the Trainer
We use TRL’s SFTTrainer (Supervised Fine-Tuning Trainer), designed specifically for instruction tuning:

from trl import SFTTrainer
from transformers import DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Set True for 5x speedup with short sequences
    args=training_args,
)
Sequence Packing

Setting packing=True can dramatically speed up training (up to 5x) for datasets with many short sequences by packing multiple examples into a single sequence. However, this can occasionally affect quality, so disable it if you notice issues.

Step 7: Train Only on Responses
A crucial optimization: we only compute loss on assistant responses, not user questions:

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
Why this matters:

User inputs provide context but don’t need to be predicted
Reduces noise in the training signal
Focuses learning on generating quality responses
More efficient use of compute
Step 8: Training Time!
Now we start the actual fine-tuning:

trainer_stats = trainer.train()
print(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f}s")
You’ll see a progress bar showing:

Current step
Training loss (should decrease over time)
Estimated time remaining
For this demo with 60 steps, training typically takes 5–15 minutes on a modern GPU. A full training run with 3 epochs on 100k examples might take several hours.

Step 9: Testing Your Model
After training, let’s test the fine-tuned model:

from transformers import TextStreamer

# Enable fast inference mode
FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Stream the response
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

outputs = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    temperature=0.7,
    min_p=0.1,
)
The FastLanguageModel.for_inference() call enables Unsloth's optimized inference mode, providing 2x speedup over standard generation.

Step 10: Saving Your Model
You have several options for saving:

Get Matteo Ferruccio Andreoni’s stories in your inbox
Join Medium for free to get updates from this writer.

Enter your email
Subscribe
Option 1: Save LoRA Adapters

model.save_pretrained("Llama32_QLoRA_fine_tuned")
tokenizer.save_pretrained("Llama32_QLoRA_fine_tuned")
This saves only the adapter weights (~100MB), which you can load later with the base model.

Option 2: Export to GGUF

For deployment with llama.cpp, Ollama, or other efficient inference engines:

model.push_to_hub_gguf(
    "Llama32_QLoRA_fine_tuned",
    tokenizer,
    quantization_method="q4_k_m"  # Good balance of size vs quality
)
Option 3: Merge and Save Full Model

model.save_pretrained_merged(
    "Llama32_merged",
    tokenizer,
    save_method="merged_16bit",  # or "lora" for adapters only
)
Performance Analysis
Let’s examine the benefits we achieved with QLoRA + Unsloth:

Memory Usage
On a typical training run:

Peak VRAM: ~6–8GB (vs 20–24GB for full fine-tuning)
Model weights: ~0.8GB (4-bit) vs 6GB (16-bit)
Optimizer states: ~0.3GB (8-bit) vs 24GB (32-bit Adam)
Training Speed
Unsloth’s optimizations provide:

2x faster than standard HuggingFace Transformers
1.5x faster than standard Unsloth without manual optimizations
30% less memory with optimized gradient checkpointing
Benchmarking Inference
import time

# Warmup
_ = model.generate(input_ids=inputs, max_new_tokens=20)

# Benchmark
start = time.time()
outputs = model.generate(input_ids=inputs, max_new_tokens=512)
duration = time.time() - start

tokens_generated = outputs.shape[1] - inputs.shape[1]
print(f"Speed: {tokens_generated / duration:.2f} tokens/sec")
Best Practices and Tips
1. Choosing LoRA Rank
Start with r=16 for most tasks. Increase to 32 or 64 for:

Complex domain adaptation
Multi-task learning
When you have abundant training data
Lower rank (8) for:

Simple tasks
Limited training data
When memory is extremely constrained
2. Learning Rate Selection
The standard 2e-4 works well for most cases, but consider:

Higher (5e-4): For small datasets or simple tasks
Lower (1e-4): For large datasets or to preserve more base model knowledge
3. Dataset Quality Over Quantity
100 high-quality examples often outperform 1000 noisy ones. Focus on:

Clear, well-formatted examples
Diverse scenarios covering your use case
Removing duplicates and low-quality data
4. Monitoring Training
Watch for:

Loss decreasing steadily: Good sign
Loss plateauing early: Try higher learning rate or more training steps
Loss unstable: Lower learning rate or check data quality
5. Preventing Overfitting
For small datasets:

Use more training epochs with lower learning rate
Monitor validation loss if you have a validation set
Consider adding dropout (though 0 is optimized)
Common Issues and Solutions
Out of Memory Errors
If you encounter OOM errors:

Reduce per_device_train_batch_size
Increase gradient_accumulation_steps proportionally
Reduce max_seq_length
Enable packing=False if it was True
Use a smaller rank (8 instead of 16)
Slow Training
To speed up:

Enable packing=True for short sequences
Ensure you’re using bfloat16 on Ampere+ GPUs
Increase batch size if memory allows
Use multiple GPUs with DDP
Poor Model Quality
If results are suboptimal:

Train for more steps/epochs
Increase LoRA rank
Check your chat template is correct
Verify data quality and formatting
Try different learning rates
Advanced Techniques
Multi-GPU Training
For multiple GPUs, simply prepend:

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
Custom Datasets
To use your own data, format it as ShareGPT conversations:

data = [
    {
        "conversations": [
            {"role": "user", "content": "Your question"},
            {"role": "assistant", "content": "Your answer"}
        ]
    }
]
Evaluation During Training
Add evaluation:

eval_dataset = load_dataset("your_eval_set")
trainer = SFTTrainer(
    # ... other args
    eval_dataset=eval_dataset,
    evaluation_strategy="steps",
    eval_steps=100,
)
Conclusion
QLoRA with Unsloth democratizes LLM fine-tuning, making it accessible to anyone with a consumer GPU.

Whether you’re a researcher, startup founder, or AI enthusiast, these techniques enable you to create custom models tailored to your specific needs without breaking the bank on compute costs.

If you liked this tutorial, check out this article too. 👇

How to Fine-Tune LLMs Locally: The Complete LoRA Guide
Master LoRA Fine-Tuning on Apple Silicon with minimal memory usage. From setup to deployment in one comprehensive…
medium.com

Resources and Next Steps
Unsloth Documentation: https://github.com/unslothai/unsloth
QLoRA Paper: arXiv:2305.14314
LoRA Paper: arXiv:2106.09685
Ready to fine-tune your first model? Clone the repository, fire up a notebook, and start experimenting!

👏 Support This Work
If you found this tutorial helpful:

Give it a clap (or 50!) to help others discover it
Follow me on Medium for more AI and machine learning tutorials
Star the GitHub repository ⭐
Share this article with your network
Have questions or suggestions? Drop them in the comments below — I read and respond to every one!

Happy fine-tuning! 🚀

