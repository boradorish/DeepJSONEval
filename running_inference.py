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
    return parser.parse_args()


def load_hf_model(model_name, hf_token=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model '{model_name}' from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        token=hf_token,
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def run_local_inference(model, tokenizer, input_message):
    import torch

    text = tokenizer.apply_chat_template(
        input_message,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    prompt_tokens = inputs['input_ids'].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=1.0,
            do_sample=False,
        )

    completion_tokens = output_ids.shape[-1] - prompt_tokens
    generated = output_ids[0][prompt_tokens:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    return response, prompt_tokens, completion_tokens


args = get_args()

benchmark_file = 'DeepJSONEval.xlsx'

benchmark_info = utils.load_excel_data(benchmark_file, 'sheet1')

to_save = benchmark_info.to_dict(orient='list')

to_save["model_output"] = []
to_save["prompt_tokens"] = []
to_save["completion_tokens"] = []

if args.use_local:
    hf_model, hf_tokenizer = load_hf_model(args.model_name, args.hf_token)

for i in tqdm(range(len(benchmark_info['schema']))):
    current_text = benchmark_info['text'][i]
    curent_schema = benchmark_info['schema'][i]
    first_half = utils.load_file(r'JSON_Output_meta_prompt.txt')
    second_half = f"*** JSON Schema\n{curent_schema}\n\n*** Text Description\n{current_text}"
    input_message = [{"role": "user", "content": first_half + '\n' + second_half}]
    try:
        if args.use_local:
            result = run_local_inference(hf_model, hf_tokenizer, input_message)
        else:
            result = utils.post_request_by_openai_format(args.base_url, args.key, args.model_name, input_message)
    except:
        result = ["Need Retry"] * 3
    to_save["model_output"].append(result[0])
    to_save["prompt_tokens"].append(result[1])
    to_save["completion_tokens"].append(result[2])

save_file_name = args.model_name.split('/')[-1].split(':')[0] + '.xlsx'
utils.save_excel_data(os.path.join(args.saving_path, save_file_name), 'sheet1', to_save)
