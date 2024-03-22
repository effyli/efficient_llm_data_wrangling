# utils for sending prompt and getting response
import ast
import openai
from utils import utils
from promptsTemplate import *

# using a iterative way to get function, which makes sure the function generated can get 100% accuracy on the provided examples
def calculate_accuracy(predicted, ground_truth):
    """
    Calculate accuracy between predicted and ground truth lists of strings.

    Args:
    predicted (list): List of predicted strings.
    ground_truth (list): List of ground truth strings.

    Returns:
    float: Accuracy percentage.
    """
    if len(predicted) != len(ground_truth):
        raise ValueError("Length of predicted and ground truth lists must be the same.")
    correct_count = 0
    for pred, truth in zip(predicted, ground_truth):
        if isinstance(pred, str) and isinstance(truth, str):
            if pred.lower() == truth.lower():
                correct_count += 1
            elif utils.evaluate_numerical_values(pred, truth):
                correct_count += 1
        else:
            if pred == truth:
                correct_count += 1
    total_count = len(predicted)

    accuracy = (correct_count / total_count) * 100
    return accuracy


def dicts_to_string(list_of_dicts):
    result = ""
    for item in list_of_dicts:
        result += f"Input: \"{item['Input']}\"\nOutput: \"{item['Output']}\"\n"
    return result


def function_calling(messages):
    """
    Building blocks for function generation
    """

    print("here is the complete messages, ", messages)
    # Define the tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "return_python",
                "description": "output of reasonining and python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "The step-by-step reason for the output",
                        },
                        "python_code": {
                            "type": "string", 
                            "description": "Python function string that transform input to output"
                        },
                    },
                    "required": ["reasoning", "python_code"],
                },
            },
        }
    ]

    client = openai.OpenAI(api_key = "Your API key here")
    response = client.chat.completions.create(
    model = 'gpt-4-turbo-preview', # Note this optional and may not be declared
    messages = messages,
    tools = tools,
    tool_choice={"type": "function", "function": {"name": "return_python"}},  # auto is default, but we'll be explicit
    stream = False,
    temperature=0.2,
    seed=42
    )
    print("here is the response", response.choices[0])
    return response

def response_parser(response):
    arguments_str = response.choices[0].message.tool_calls[0].function.arguments
    # check some conditions
    arguments_dict = ast.literal_eval(arguments_str)
    return arguments_dict['python_code']

def formulate_prompt(instruction: str, examples: str, task: str="data_transformation"):
    if task == "data_transformation":
        template_name = STRING_TRANSFORMATION
    elif task == "entity_matching":
        template_name = ENTITY_MATCHING
    elif task == "data_imputation":
        template_name = DATA_IMPUTATION
    elif task == "error_detection_spelling":
        template_name = ERROR_DETECTION_SPELLING
    else:
        raise NotImplementedError
    prefix = TASK_PREFIX
    template = template_name
    template_name = prefix + template
    prompt_template = PromptTemplate(prompt=template_name)
    prompts = []
    for prompt_t in prompt_template.prompt:
        prompt = prompt_t.copy()
        if 'user' in prompt_t['role']:
            print("Found user")
            prompt['content'] = prompt['content'].format(instruction=instruction, examples=examples)
        prompts.append(prompt)
    return prompts, prompt_template


def execute_function_string(fn, input_string, task):
    # Prepare a custom namespace for executing the dynamic code. This can be an empty dict.
    if task == "data_transformation":
        fn_name = "string_transformation"
    elif task in ["entity_matching", "data_imputation"]:
        fn_name = "input_output_converter"
    elif task == "error_detection_spelling":
        fn_name = "detect_error"
    else:
        raise NotImplementedError
    namespace = {}
    # Execute the dynamic code. This will define the function within the 'namespace'.
    exec(fn, namespace)
    # Access the function from the namespace and call it
    func = namespace[fn_name]
    result = func(input_string) 
    return result


def evaluate(fn, test_data, task):
    inputs = [item["Input"] for item in test_data]
    outputs = [item["Output"] for item in test_data]
    predicted_outputs = []
    for input, output in zip(inputs, outputs):
        try:
            result = execute_function_string(fn, input, task)
        except Exception as error:
            print(f"Error message: {error}")
            result = "Not excutable"
        predicted_outputs.append(result)
        print("pred: {}, gt: {}".format(result, output))
    metrics = utils.compute_metrics(preds=predicted_outputs, golds=outputs, task=task)
    if task in [
            "data_imputation",
            "data_transformation",
            
        ]:
        acc = metrics[2]
    elif task in ["entity_matching", "error_detection_spelling"]:
        acc = metrics[3]
    print("The accuacry/f1 of the generated function is ", acc)
    return acc, predicted_outputs


def is_excutable(fn, test_data, task):
    try:
        _ = execute_function_string(fn, test_data[0]['Input'], task)
    except Exception as e:
        return False, f"Error: {str(e)}"
    return True, "Execution successful."
    

def validate_function(fn, test_data, task, supervision_data, threshold_sup=10., threshold=0.51):
    """
    We validate function by checking:
    1. If the function runs.
    2. If the function get ~100% accuracy on the demonstrations.
    3. If the function is generalizable by prompting LLMs.
    """
    runnable, msg = is_excutable(fn, test_data, task)
    if not runnable:
        error_message = f"The generated function {fn} is not excutable, error message is {msg}. Please fix the function."
        return "not_excutable", error_message, 0
    acc, _ = evaluate(fn, test_data, task)
    # we don't expect accuracy to be 100% on currency exchange tasks
    if acc < threshold:
        error_message = f"The generated function {fn} only achieves {acc} accuracy on provided examples. Please provide a different function to achieve higher accuracy."
        return "not_fit", error_message, acc
    if supervision_data and not len(supervision_data) == 0:
        acc_sup, _ = evaluate(fn, supervision_data, task)
        if acc_sup < threshold_sup:
            error_message = f"The generated function {fn} is overfitting on the provided example, please reconsider the intention of this task."
            return "not_generalizable", error_message, acc
    return "validated", "", acc


def generate_function_pipeline(instruction: str, examples: list, task: str, supervision_data=None, depth: int=5):
    """
    The sequence of action for generating the final function is: generate function -> validate function -> correct function if necessary.
    """
    # Define the messages
    messages, _ = formulate_prompt(instruction, dicts_to_string(examples), task=task)   
    func = None
    while not func:
        response = function_calling(messages)
        try:
            func = response_parser(response)
        except:
            print("Function calling return not parsable response, trying again!")
            func = None
            continue
    print(func)
    num_iter = 1
    functions_stack = []
    while num_iter < depth:
        decision, msg, acc = validate_function(func, examples, task, supervision_data)
        if "not" in decision:
            if not decision == "not_excutable":
                functions_stack.append((func, acc))
            # if this is the last try and the function runs then we return
            if (num_iter == depth - 1):
                if functions_stack:
                    # sort the function stack
                    sorted_func_stack = sorted(functions_stack, key = lambda x: x[1], reverse=True)
                    return sorted_func_stack[0]
            # add error message and re-prompt
            retry_msg = TASK_RETRY[0].copy()
            retry_msg['content'] = retry_msg['content'].format(error_message=msg)
            messages.append(retry_msg)
            response = function_calling(messages)
            func = response_parser(response)
        else:
            return (func, acc)
        num_iter += 1
    return "No function can be generated, please provide different demonstration!", float("-inf")




    
