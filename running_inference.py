import utils
from tqdm import tqdm
import os
import argparse
import shutil

# temperature 0.6
# python running_inference.py --use-local --model-name boradorish/qwen3-0.6b-fc --temperature 0.6

# temperature 1.0
# python running_inference.py --use-local --model-name Qwen/Qwen3-8B --temperature 1.0


def get_args():
    parser = argparse.ArgumentParser('DeepJSON inference script')
    parser.add_argument('--base-url', default='', type=str, help="base url of LLM chat api")
    parser.add_argument('--key', default='', type=str, help='api key for using the LLM chat api')
    parser.add_argument('--model-name', default='', type=str, help='name of model when post request to the LLM chat api')
    parser.add_argument('--saving-path', default='', type=str, help='the path of folder in which the inference result file locates')
    parser.add_argument('--use-local', action='store_true', help='load model locally using vLLM instead of using API')
    parser.add_argument('--hf-token', default=None, type=str, help='HuggingFace access token for private models')
    parser.add_argument('--base-model', default='Qwen/Qwen3-0.6B', type=str, help='base model name for tokenizer fallback')
    parser.add_argument('--thinking-budget', default=1024, type=int, help='max tokens for Qwen3 thinking (0 to disable)')
    parser.add_argument('--temperature', default=0.6, type=float, help='sampling temperature for vLLM inference (try 0.6 or 1.0)')
    parser.add_argument('--num-runs', default=1, type=int, help='number of inference runs (model loaded once, results saved separately)')
    parser.add_argument('--max-model-len', default=None, type=int, help='maximum model context length (reduce if GPU OOM, e.g. 8192)')
    parser.add_argument('--cc', default=None, type=str, help='C compiler path for Triton/vLLM; only sets CC for this process')
    return parser.parse_args()


def configure_vllm_environment(hf_token=None):
    # Must be set before importing vllm.
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token


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


def build_vllm_kwargs(max_model_len=None):
    kwargs = {
        'dtype': 'float16',
        'trust_remote_code': True,
        'enforce_eager': True,
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
    hf_token=None,
    base_model_name='Qwen/Qwen3-0.6B',
    max_model_len=None,
    cc_path=None,
):
    configure_vllm_environment(hf_token)
    ensure_c_compiler_available(cc_path)

    from vllm import LLM

    kwargs = build_vllm_kwargs(max_model_len)

    print(f"Loading model '{model_name}' with vLLM...")
    try:
        llm = LLM(model=model_name, tokenizer=model_name, **kwargs)
    except Exception as error:
        if not is_tokenizer_error(error):
            raise

        print(f"Tokenizer failed for '{model_name}', falling back to base model '{base_model_name}'...")
        llm = LLM(model=model_name, tokenizer=base_model_name, **kwargs)

    print("Model loaded.")
    return llm


def run_vllm_inference_batch(llm, messages_batch, temperature=0.6, thinking_budget=1024):
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()

    texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, thinking_budget=thinking_budget)
        for msg in messages_batch
    ]

    sampling_params = SamplingParams(temperature=temperature, max_tokens=4096)
    outputs = llm.generate(texts, sampling_params)
    # huggingface inference 쓰지 말고 vllm 에서 llm.generate() 함수 활용해서 구현해
    # temperature 0.6 이랑 1.0 둘다 써보고 성능 어떻게 나오는지 보기
    # error analysis : 1) schema 안 맞춤, 2) type 안 맞춤
        # > 데이터셋 문제인가요?
        # 1) 모델 크기 크기가 너무 작나? > 8B 평가중임
        # - 데이터셋 제작은 잠깐 보류
        # 2) 8B 결과 알려주세요
        # 3) SFT 보다 Preference Optimization 이 나을 수도
        # X (Y_w > Y_l)

    results = []
    for output in outputs:
        response = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        results.append((response, prompt_tokens, completion_tokens))

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

    benchmark_file = 'DeepJSONEval.xlsx'
    benchmark_info = utils.load_excel_data(benchmark_file, 'sheet1')
    to_save = benchmark_info.to_dict(orient='list')

    vllm_model = None
    if args.use_local:
        vllm_model = load_vllm_model(
            args.model_name,
            args.hf_token,
            args.base_model,
            args.max_model_len,
            args.cc,
        )

    all_messages = build_messages(benchmark_info)
    num_runs = args.num_runs if args.use_local else 1

    for run_idx in range(num_runs):
        run_label = f" (run {run_idx + 1}/{num_runs})" if num_runs > 1 else ""
        print(f"\nStarting inference{run_label}...")

        reset_output_columns(to_save)

        if args.use_local:
            try:
                results = run_vllm_inference_batch(vllm_model, all_messages, args.temperature, args.thinking_budget)
            except Exception as error:
                print(f"vLLM inference failed: {error}")
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
