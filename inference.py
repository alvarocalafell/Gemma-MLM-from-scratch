from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str) -> dict:
    """
    Move model inputs to the specified device.

    Args:
        model_inputs (dict): A dictionary containing model input tensors.
        device (str): The target device to move the inputs to (e.g., 'cuda', 'cpu').

    Returns:
        dict: A dictionary with all input tensors moved to the specified device.
    """
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
) -> dict:
    """
    Prepare model inputs from a prompt and an image file.

    Args:
        processor (PaliGemmaProcessor): The processor for tokenizing text and processing images.
        prompt (str): The text prompt to process.
        image_file_path (str): The file path of the image to process.
        device (str): The device to move the processed inputs to.

    Returns:
        dict: A dictionary containing the processed and device-moved model inputs.
    """
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> None:
    """
    Run inference on the model with the given inputs and parameters.

    Args:
        model (PaliGemmaForConditionalGeneration): The model to run inference on.
        processor (PaliGemmaProcessor): The processor for tokenizing and processing inputs.
        device (str): The device to run inference on.
        prompt (str): The text prompt to use for generation.
        image_file_path (str): The file path of the image to use.
        max_tokens_to_generate (int): The maximum number of tokens to generate.
        temperature (float): The temperature for sampling.
        top_p (float): The top-p value for nucleus sampling.
        do_sample (bool): Whether to use sampling for generation.

    Returns:
        None: Prints the generated text.
    """
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)


def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on the given probability distribution.

    Args:
        probs (torch.Tensor): The input probability distribution.
        p (float): The cumulative probability threshold for top-p sampling.

    Returns:
        torch.Tensor: The sampled token indices.
    """
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
) -> None:
    """
    Main function to load the model and run inference.

    Args:
        model_path (str, optional): Path to the model. Defaults to None.
        prompt (str, optional): Text prompt for generation. Defaults to None.
        image_file_path (str, optional): Path to the image file. Defaults to None.
        max_tokens_to_generate (int, optional): Maximum number of tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 0.8.
        top_p (float, optional): Top-p value for nucleus sampling. Defaults to 0.9.
        do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
        only_cpu (bool, optional): Whether to force CPU usage only. Defaults to False.

    Returns:
        None: Runs the inference and prints the result.
    """
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print("Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)