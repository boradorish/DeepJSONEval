import json
import pandas as pd
from openai import OpenAI
from jsonschema import validate


def load_file(file_path):
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = file.read()
    return data


def load_excel_data(data_path, sheet):
    data = pd.read_excel(data_path, sheet_name=sheet)
    return data


def save_excel_data(data_path, sheet, data):
    writer = pd.ExcelWriter(data_path)
    data = pd.DataFrame(data)
    data.to_excel(writer, sheet_name=sheet)
    writer.close()
    return


def post_request_by_openai_format(input_base_url, key, model_name, input_message):
    client = OpenAI(
        base_url=input_base_url,
        api_key=key
    )

    completion = client.chat.completions.create(
        extra_headers={},
        extra_body={},
        model=model_name,
        messages=input_message,
        temperature=0
    )

    return completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens


'''
you can add your own method of calling LLM inference API and replace the method in running_inference.py or running_inference_continue.py
'''

def compare_values(answer, model_output):
    # base case
    if isinstance(answer, (str, bool, int, float)):
            return 1 if answer == model_output else 0

    # list case
    elif isinstance(answer, list):
        # if list is empty, return 1 when model output is also empty
        if not answer:
            return 1 if (isinstance(model_output, list) and not model_output) else 0

        # if list of dict, compare index by index
        if all(isinstance(item, dict) for item in answer):
            if not isinstance(model_output, list) or not all(isinstance(item, dict) for item in model_output):
                return 0
            
            score = 0
            min_len = min(len(answer), len(model_output))
            max_len = max(len(answer), len(model_output)) ## as it is hard for list of dict to calculate union, use max length to substitute

            for i in range(min_len):
                score += compare_values(answer[i], model_output[i])

            return score / max_len if (max_len > 0) else 1
        
        # if list of base data types, compute Jaccard similarity
        else:
            if not isinstance(model_output, list):
                return 0
            
            answer_set = set(answer)
            model_output_set = set(model_output)

            common_elements = answer_set & model_output_set
            all_elements = answer_set | model_output_set

            return len(common_elements) / len(all_elements) if all_elements else 1

    # dict case   
    elif isinstance(answer, dict):
        if not isinstance(model_output, dict):
            return 0
        
        answer_keys = set(answer.keys())
        if not answer_keys: # empty dict
                return 1 if not model_output else 0
        
        all_keys = answer_keys.union(set(model_output.keys()))

        score = 0
        for key in answer_keys:
            if key in model_output:
                # compare the value in common keys recursively
                score += compare_values(answer[key], model_output[key])

        return score / len(all_keys) if all_keys else 1
    
    # other data types 
    else:
        return 0


def remove_title_fields(value):
    if isinstance(value, dict):
        return {
            key: remove_title_fields(item)
            for key, item in value.items()
            if key != "title"
        }

    if isinstance(value, list):
        return [remove_title_fields(item) for item in value]

    return value


def remove_title_from_schema(schema):
    if isinstance(schema, dict):
        cleaned_schema = {}
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                cleaned_schema[key] = {
                    property_key: remove_title_from_schema(property_value)
                    for property_key, property_value in value.items()
                    if property_key != "title"
                }
            elif key == "required" and isinstance(value, list):
                cleaned_schema[key] = [item for item in value if item != "title"]
            else:
                cleaned_schema[key] = remove_title_from_schema(value)
        return cleaned_schema

    if isinstance(schema, list):
        return [remove_title_from_schema(item) for item in schema]

    return schema



def extract_json_from_output(model_output: str):
    import re

    if not isinstance(model_output, str):
        return None

    # Qwen3 thinking mode: </think> 이후만 파싱
    if "</think>" in model_output:
        model_output = model_output.split("</think>")[-1].strip()

    # 1. ```json ... ```
    if "```json" in model_output:
        candidate = model_output.split("```json")[-1].split("```")[0].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 2. ``` ... ```
    if "```" in model_output:
        candidate = model_output.split("```")[1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3. 텍스트에서 { ... } 또는 [ ... ] 추출
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        match = re.search(pattern, model_output)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

    # 4. 출력 자체가 JSON
    try:
        return json.loads(model_output.strip())
    except Exception:
        pass

    return None


def json_evaluation_new(model_output: str, answer: str, schema: dict):
    model_output_json = extract_json_from_output(model_output)
    if model_output_json is None:
        return 0, 0, 0, "No valid JSON found in model output"

    try:
        model_output_json = json.loads(model_output_json) if isinstance(model_output_json, str) else model_output_json
    except:
        return 0, 0, 0, "Not a invalid JSON"
    
    answer_json = json.loads(answer)
    model_output_json = remove_title_fields(model_output_json)
    answer_json = remove_title_fields(answer_json)
    schema = remove_title_from_schema(schema)

    try:
        validate(instance=model_output_json, schema=schema)
    except:
        return 0, 0, 0, "JSON output doesn't match the schema"
    
    format_score = 1

    if model_output_json == answer_json:
        strict_score = 1
    else:
        strict_score = 0

    similarity_score = compare_values(answer_json, model_output_json)

    return format_score, similarity_score, strict_score, "Give score in 3 criteria"
