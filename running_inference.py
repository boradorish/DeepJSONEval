import utils
from tqdm import tqdm
import os
import argparse
import shutil

# temperature 0.6
# python running_inference.py --use-local --model-name ../LLaMA-Factory/saves/qwen3-0.6b/full/dpo --temperature 0.6

# temperature 1.0
# python running_inference.py --use-local --model-name Qwen/Qwen3-8B --temperature 1.0


def get_args():
    parser = argparse.ArgumentParser('DeepJSON inference script')
    parser.add_argument('--base-url', default='', type=str, help="base url of LLM chat api")
    parser.add_argument('--key', default='', type=str, help='api key for using the LLM chat api')
    parser.add_argument('--model-name', default='', type=str, help='name of model when post request to the LLM chat api')
    parser.add_argument('--tokenizer', default=None, type=str, help='tokenizer path/name for vLLM local inference')
    parser.add_argument('--saving-path', default='', type=str, help='the path of folder in which the inference result file locates')
    parser.add_argument('--use-local', action='store_true', help='load model locally using vLLM instead of using API')
    parser.add_argument('--backend', choices=('vllm', 'transformers'), default='vllm', help='local inference backend used with --use-local')
    parser.add_argument('--hf-token', default=None, type=str, help='HuggingFace access token for private models')
    parser.add_argument('--base-model', default='Qwen/Qwen3-4B', type=str, help='base model name for tokenizer fallback')
    parser.add_argument('--thinking-budget', default=0, type=int, help='max tokens for Qwen3 thinking (0 disables thinking)')
    parser.add_argument('--temperature', default=0.6, type=float, help='sampling temperature for vLLM inference (try 0.6 or 1.0)')
    parser.add_argument('--top-p', default=0.95, type=float, help='nucleus sampling top-p for local inference')
    parser.add_argument('--max-new-tokens', default=1024, type=int, help='maximum new tokens generated per prompt')
    parser.add_argument('--num-runs', default=1, type=int, help='number of inference runs (model loaded once, results saved separately)')
    parser.add_argument('--max-model-len', default=None, type=int, help='maximum model context length (reduce if GPU OOM, e.g. 8192)')
    parser.add_argument('--gpu-ids', default=None, type=str, help='comma-separated GPU ids to expose, e.g. 0 or 0,1; sets CUDA_VISIBLE_DEVICES before loading vLLM')
    parser.add_argument('--tensor-parallel-size', default=1, type=int, help='number of GPUs used by vLLM tensor parallelism')
    parser.add_argument('--gpu-memory-utilization', default=0.9, type=float, help='fraction of visible GPU memory vLLM may reserve')
    parser.add_argument('--attention-backend', default=None, type=str, help='optional vLLM attention backend, e.g. FLASH_ATTN, XFORMERS, FLASHINFER')
    parser.add_argument('--cc', default=None, type=str, help='C compiler path for Triton/vLLM; only sets CC for this process')
    return parser.parse_args()


def configure_vllm_environment(hf_token=None, attention_backend=None):
    # Must be set before importing vllm.
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token


def ensure_c_compiler_available(cc_path=None):
    if cc_path:
        resolved_cc_path = shutil.which(cc_path) or cc_path
        if os.path.exists(resolved_cc_path):
            os.environ["CC"] = resolved_cc_path
            return
        raise RuntimeError(f"Requested C compiler was not found: {cc_path}")

    if os.environ.get("CC"):
        return

    compiler_candidates = (
        "cc",
        "gcc",
        "clang",
        "x86_64-conda-linux-gnu-gcc",
        "aarch64-conda-linux-gnu-gcc",
    )

    for compiler in compiler_candidates:
        compiler_path = shutil.which(compiler)
        if compiler_path:
            os.environ["CC"] = compiler_path
            return

    raise RuntimeError(
        "vLLM/Triton requires a C compiler, but none was found. "
        "On a shared server, install a compiler in your own conda environment "
        "instead of changing the system packages, then pass it via --cc or CC. "
        "Example: conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y "
        "&& python running_inference.py --use-local --cc x86_64-conda-linux-gnu-gcc ..."
    )


def build_vllm_kwargs(max_model_len=None, tensor_parallel_size=1, gpu_memory_utilization=0.9):
    kwargs = {
        'trust_remote_code': True,
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': gpu_memory_utilization,
    }

    if max_model_len is not None:
        kwargs['max_model_len'] = max_model_len
    return kwargs


def is_tokenizer_error(error):
    message = str(error).lower()
    return "tokenizer" in message and any(
        keyword in message
        for keyword in ("not found", "failed", "could not", "couldn't", "unable")
    )


def load_vllm_model(
    model_name,
    tokenizer_name=None,
    hf_token=None,
    base_model_name='Qwen/Qwen3-0.6B',
    max_model_len=None,
    cc_path=None,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    attention_backend=None,
):
    configure_vllm_environment(hf_token, attention_backend)
    ensure_c_compiler_available(cc_path)

    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer_name = tokenizer_name or model_name
    tokenizer_kwargs = {'trust_remote_code': True}
    if hf_token:
        tokenizer_kwargs['token'] = hf_token

    print(f"Loading tokenizer '{tokenizer_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    except Exception as error:
        if not is_tokenizer_error(error):
            raise
        print(f"Tokenizer failed for '{tokenizer_name}', falling back to base model '{base_model_name}'...")
        tokenizer_name = base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    kwargs = build_vllm_kwargs(max_model_len, tensor_parallel_size, gpu_memory_utilization)
    if attention_backend:
        kwargs['attention_backend'] = attention_backend
    if hf_token:
        kwargs['hf_token'] = hf_token

    print(f"Loading model '{model_name}' with vLLM...")
    llm = LLM(model=str(model_name), tokenizer=str(tokenizer_name), **kwargs)

    print("Model loaded.")
    return llm, tokenizer


def load_transformers_model(
    model_name,
    hf_token=None,
    base_model_name='Qwen/Qwen3-0.6B',
):
    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16,
        'token': hf_token,
    }

    print(f"Loading model '{model_name}' with Transformers...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    except Exception as error:
        if not is_tokenizer_error(error):
            raise
        print(f"Tokenizer failed for '{model_name}', falling back to base model '{base_model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=hf_token)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', **load_kwargs)
    except (ImportError, ValueError) as error:
        if "accelerate" not in str(error).lower():
            raise
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"accelerate is not available; loading model on {device} without device_map='auto'.")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def apply_chat_template(tokenizer, messages, thinking_budget=0):
    try:
        kwargs = {
            'tokenize': False,
            'add_generation_prompt': True,
            'enable_thinking': thinking_budget > 0,
        }
        if thinking_budget > 0:
            kwargs['thinking_budget'] = thinking_budget
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_vllm_inference_batch(
    llm_and_tokenizer,
    messages_batch,
    temperature=0.6,
    top_p=0.95,
    max_new_tokens=1024,
    thinking_budget=0,
):
    from vllm import SamplingParams

    llm, tokenizer = llm_and_tokenizer

    texts = [
        apply_chat_template(tokenizer, msg, thinking_budget)
        for msg in messages_batch
    ]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    print("Starting one vLLM offline generate call...")
    outputs = llm.generate(texts, sampling_params)
    print("Generation done.")

    results = []
    for output in outputs:
        response = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        results.append((response, prompt_tokens, completion_tokens))

    return results


def run_transformers_inference_batch(model_and_tokenizer, messages_batch, temperature=0.6, thinking_budget=0):
    import torch

    model, tokenizer = model_and_tokenizer
    results = []
    model_device = next(model.parameters()).device

    for messages in tqdm(messages_batch):
        text = apply_chat_template(tokenizer, messages, thinking_budget)
        inputs = tokenizer(text, return_tensors='pt')
        prompt_tokens = int(inputs['input_ids'].shape[-1])
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = generated_ids[0][prompt_tokens:]
        response = tokenizer.decode(completion_ids, skip_special_tokens=True)
        results.append((response, prompt_tokens, int(completion_ids.shape[-1])))

    return results


def build_messages(benchmark_info):
    first_half = utils.load_file(r'JSON_Output_meta_prompt.txt')
    messages = []

    for i in range(len(benchmark_info['schema'])):
        current_text = benchmark_info['text'][i]
        current_schema = benchmark_info['schema'][i]
        second_half = f"*** JSON Schema\n{current_schema}\n\n*** Text Description\n{current_text}"
        messages.append([{"role": "user", "content": first_half + '\n' + second_half}])

    return messages


def reset_output_columns(to_save):
    to_save["model_output"] = []
    to_save["prompt_tokens"] = []
    to_save["completion_tokens"] = []


def save_results(args, to_save, run_idx, num_runs):
    temp_str = f"_t{args.temperature}" if args.use_local else ""
    run_str = f"_run{run_idx + 1}" if num_runs > 1 else ""
    save_file_name = args.model_name.split('/')[-1].split(':')[0] + temp_str + run_str + '.xlsx'
    utils.save_excel_data(os.path.join(args.saving_path, save_file_name), 'sheet1', to_save)
    print(f"Saved: {save_file_name}")


def main():
    args = get_args()

    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    benchmark_file = 'DeepJSONEval.xlsx'
    benchmark_info = utils.load_excel_data(benchmark_file, 'sheet1')
    to_save = benchmark_info.to_dict(orient='list')

    local_model = None
    if args.use_local:
        if args.backend == 'vllm':
            local_model = load_vllm_model(
                args.model_name,
                args.tokenizer,
                args.hf_token,
                args.base_model,
                args.max_model_len,
                args.cc,
                args.tensor_parallel_size,
                args.gpu_memory_utilization,
                args.attention_backend,
            )
        else:
            local_model = load_transformers_model(args.model_name, args.hf_token, args.base_model)

    all_messages = build_messages(benchmark_info)
    num_runs = args.num_runs if args.use_local else 1

    for run_idx in range(num_runs):
        run_label = f" (run {run_idx + 1}/{num_runs})" if num_runs > 1 else ""
        print(f"\nStarting inference{run_label}...")

        reset_output_columns(to_save)

        if args.use_local:
            try:
                if args.backend == 'vllm':
                    results = run_vllm_inference_batch(
                        local_model,
                        all_messages,
                        args.temperature,
                        args.top_p,
                        args.max_new_tokens,
                        args.thinking_budget,
                    )
                else:
                    results = run_transformers_inference_batch(local_model, all_messages, args.temperature, args.thinking_budget)
            except Exception as error:
                print(f"{args.backend} inference failed: {error}")
                results = [("Need Retry", 0, 0)] * len(all_messages)

            for result in results:
                to_save["model_output"].append(result[0])
                to_save["prompt_tokens"].append(result[1])
                to_save["completion_tokens"].append(result[2])
        else:
            for i in tqdm(range(len(all_messages))):
                try:
                    result = utils.post_request_by_openai_format(args.base_url, args.key, args.model_name, all_messages[i])
                except Exception as error:
                    print(f"API inference failed at row {i}: {error}")
                    result = ["Need Retry"] * 3
                to_save["model_output"].append(result[0])
                to_save["prompt_tokens"].append(result[1])
                to_save["completion_tokens"].append(result[2])

        save_results(args, to_save, run_idx, num_runs)


if __name__ == "__main__":
    main()
