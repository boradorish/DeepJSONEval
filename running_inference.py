import utils
from tqdm import tqdm
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser('DeepJSON inference script')
    parser.add_argument('--base-url', default='', type=str, help="base url of LLM chat api")
    parser.add_argument('--key', default='', type=str, help='api key for using the LLM chat api')
    parser.add_argument('--model-name', default='', type=str, help='name of model when post request to the LLM chat api')
    parser.add_argument('--saving-path', default='', type=str, help='the path of folder in which the inference result file locates')
    parser.add_argument('--use-local', action='store_true', help='load model locally from HuggingFace instead of using API')
    parser.add_argument('--hf-token', default=None, type=str, help='HuggingFace access token for private models')
    parser.add_argument('--base-model', default='Qwen/Qwen3-0.6B', type=str, help='base model name for tokenizer fallback')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size for local model inference')
    parser.add_argument('--thinking-budget', default=1024, type=int, help='max tokens for Qwen3 thinking (0 to disable)')
    return parser.parse_args()


def load_hf_model(model_name, hf_token=None, base_model_name='Qwen/Qwen3-0.6B'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model '{model_name}' from HuggingFace...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except OSError:
        print(f"Tokenizer not found in '{model_name}', falling back to base model '{base_model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        token=hf_token,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def run_local_inference_batch(model, tokenizer, messages_batch, thinking_budget=1024):
    import torch

    texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, thinking_budget=thinking_budget)
        for msg in messages_batch
    ]

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(model.device)
    input_len = inputs['input_ids'].shape[1]
    prompt_lengths = inputs['attention_mask'].sum(dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    results = []
    for i in range(len(messages_batch)):
        generated = output_ids[i][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        prompt_tokens = int(prompt_lengths[i])
        completion_tokens = int(generated.shape[0])
        results.append((response, prompt_tokens, completion_tokens))

    return results


args = get_args()

benchmark_file = 'DeepJSONEval.xlsx'

benchmark_info = utils.load_excel_data(benchmark_file, 'sheet1')

to_save = benchmark_info.to_dict(orient='list')

to_save["model_output"] = []
to_save["prompt_tokens"] = []
to_save["completion_tokens"] = []

if args.use_local:
    hf_model, hf_tokenizer = load_hf_model(args.model_name, args.hf_token, args.base_model)

first_half = utils.load_file(r'JSON_Output_meta_prompt.txt')

all_messages = []
for i in range(len(benchmark_info['schema'])):
    current_text = benchmark_info['text'][i]
    curent_schema = benchmark_info['schema'][i]
    second_half = f"*** JSON Schema\n{curent_schema}\n\n*** Text Description\n{current_text}"
    all_messages.append([{"role": "user", "content": first_half + '\n' + second_half}])

if args.use_local:
    for i in tqdm(range(0, len(all_messages), args.batch_size)):
        batch = all_messages[i:i + args.batch_size]
        try:
            results = run_local_inference_batch(hf_model, hf_tokenizer, batch, args.thinking_budget)
        except:
            results = [("Need Retry", 0, 0)] * len(batch)
        for result in results:
            to_save["model_output"].append(result[0])
            to_save["prompt_tokens"].append(result[1])
            to_save["completion_tokens"].append(result[2])
else:
    for i in tqdm(range(len(all_messages))):
        try:
            result = utils.post_request_by_openai_format(args.base_url, args.key, args.model_name, all_messages[i])
        except:
            result = ["Need Retry"] * 3
        to_save["model_output"].append(result[0])
        to_save["prompt_tokens"].append(result[1])
        to_save["completion_tokens"].append(result[2])

save_file_name = args.model_name.split('/')[-1].split(':')[0] + '.xlsx'
utils.save_excel_data(os.path.join(args.saving_path, save_file_name), 'sheet1', to_save)
