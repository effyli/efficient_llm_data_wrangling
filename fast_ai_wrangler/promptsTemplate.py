from typing import *


TASK_PREFIX = [ {"role": "system", "content": "You are helpful assistant. Given the following input/output examples, generate a python function that converts input to output. Reason step-by-step about the what the function is supposed to do and then generate the function. You can assume that the following python libraries are available in the target environment."}]

TASK_RETRY = [ {"role": "user", "content": f"{{error_message}}"}]

STRING_TRANSFORMATION = [{"role": "user", "content": """You are an expert Python user. I will give you a string transformation task. \
                         The task involves converting input strings to output strings. \
                         Assuming all of the following python packages are installed: ["regex", "fractions", "math", "pyproj", "BeautifulSoup", "geopy", "ummalqura", "mgrs", "pytz", "datetime", "calendar", "roman"],\
                          you can **import** them and please put all imports in the begining of the generated python program. You can also import other packages if the listed ones are not working, but try using listed packages first. \
                          When it comes to currency exchange, please do not use real-time currency, but figure out the exchange rate based on the examples I provide.
                         You will have to find the Python program called "string_transformation" that will generate the output string given the input string. \
                          Output can only be string and nothing else. \
                          Please try to understand the intentions of the input-output examples while generating the function. \
                          Please return only the python function and nothing else.  Include any imports. Do not provide any comments.\
                          Here is a very simple example: \n
                          Input: "john.doe@example.com" \n 
                          Output: "Yes" \n 
                          Input: "jane_doe@example.com" \n 
                          Output: "Yes" 
                          Input: "example.com" \n
                          Output: "No" \n
                          Generated function string: \n
                          ```pythonimport re\ndef string_transformation(input_string):\n\tpattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'\n\tif re.match(pattern, input_string):\n\t\treturn "Yes"\n\telse:
        return\n\t\t"No"```\n
                          Now it's your turn. After generating the function string, please add escape symbol tab '\t' to format the code properly with indents like shown in the example. """},\
                        {"role": "user", "content": f"""Instructions: {{instruction:}}, Examples: {{examples:}}"""}]

ENTITY_MATCHING = [{"role": "user", "content": """I have an entity matching task to solve. I have pairs of rows and I want you to find a python program that checks if they match with each other. 
                    Please try to match using substring and regex preferably. 
                    First consider using Jaccard, levenshtein, cosine similarity or any other edit distance function to solve the problem. 
                    Try to define a low threshold so that the generated function is **generalizable**.
                    Note that there can be missing values in some columns, and you can skip them for matching. All columns are connected by the symbol ^ in the input string.
                    Please do not train any machine learning classifier.
                    You can try to choose appropriate distance function with revevant columns based on the column meaning.
                    \n
                    Assuming all of the following python packages are installed in the target environment: ["numpy", "regex", "fractions", "sklearn", "Levenshtein", "math", "pyproj", "BeautifulSoup", "geopy", "ummalqura", "mgrs", "pytz", "datetime", "calendar", "roman"], 
                    you can import them when it is necessary and please put all imports in the beginning of the generated python program. 
                    You can also import other packages if the listed ones are not working, but try using listed packages first. 
                    If you need to list some mappings, try to retrieve all entries first or write a function to retrieve it first then write the matching function. 
                    Try to generate functions that can generalize to many other data samples than the ones we provided here. 
                    \n
                    The main python function that's callable to solve the task is called "input_output_converter(input_string)" where "input_string" is the function parameter. 
                    One example function string is: ```pythonimport re\ndef input_output_converter(input_string):\n\tpattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'\n\tif re.match(pattern, input_string):\n\t\treturn "Yes"\n\telse: return\n\t\t"No"``` 
                    \n
                    Please return only the python function and nothing else. Include any imports. Please do **not** provide any comments. 
                    After generating the function string, please add escape symbol tab '\t' to format the code properly with indents like shown in the example."""},\
                        {"role": "user", "content": f"""Instructions: {{instruction:}}, Examples: {{examples:}}"""}]


DATA_IMPUTATION = [{"role": "user", "content": """You are a helpful code searcher. I have a data imputation task to solve. I have rows of records and I want you to find a python program that substituting missing values given the column name.

                    Please try to find programs that uses logical conditions, substring and regex preferably, in combination of the column meanings. 
                    Try to define the range dynamically so that the generated program is **generalizable**. 
                    Please do not train any machine learning classifier.

                    Assuming all of the following python packages are installed in the target environment: ["regex", "fractions", "sklearn", "Levenshtein", "math", "pyproj", "BeautifulSoup", "geopy", "ummalqura", "mgrs", "pytz", "datetime", "calendar", "roman"], you can import them when it is necessary and please put all imports in the beginning of the generated python program. 
                    You can also import other packages if the listed ones are not working, but try using listed packages first. 
                    If you need to list some mappings, try to retrieve all entries first or write a function to retrieve it first then write the matching function. 
                    Try to generate functions that can generalize to many other data samples than the ones we provided here.
                    The main python function that's callable to solve the task is called "input_output_converter(input_string)" where "input_string" is the function parameter. 
                    One example function string is: pythonimport re\ndef input_output_converter(input_string):\n\tpattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'\n\tif re.match(pattern, input_string):\n\t\treturn "Yes"\n\telse: return\n\t\t"No"

                    Please return only the python function and nothing else. Include any imports. Please do not provide any comments. After generating the function string, please add escape symbol tab '\t' to format the code properly with indents like shown in the example.

                    """},
                    {"role": "user", "content": f"""Instructions: {{instruction:}}, Examples: {{examples:}}"""}]

ERROR_DETECTION_SPELLING = [{"role": "user", "content": """You are a helpful code generator. I have an error detection task to solve. I have rows of records from a table that contains only one column, and I want you to find a python program that detects if there is an error in the column. 
                             The error are often spelling errors, where the unusual tokens are introduced in the phrase. Hence, please try to avoid exact match in the program so it can **generalize** to other data samples.

                            The main python function that's callable to solve the task is called "detect_error(input_string)" where "input_string" is the function parameter, which often is the column value. 
                            If you need to import the python packages, assume they are already installed in the target environment. 
                            Assuming the list of python packages are installed: ["SpellChecker", "regex", "nltk", "fractions", "sklearn", "Levenshtein", "math"].
                            If you need to use spellchecker, then please try to include domain-specific vocabularies.
                            Please find functions that can generalize to many other tables of the same task. Please avoid listing exact match to the words in the examples in your generated algorithm. 
                            Please return only the python function and nothing else. Include any imports. Please do not provide any comments. After generating the function string, please add escape symbol tab '\t' to format the code properly with indents.

                            I will provide you some input/output examples as references.
                            """},
                            {"role": "user", "content": f"""Instructions: {{instruction:}}, Examples: {{examples:}}"""}]

ERROR_DETECTION_SPELLING_INSTR = f"""I have a table with adult data, now I will provide rows of records with the column name: {{column_name}}, please write an appropriate python program that checks if there are any spelling errors. Return "yes" if there is spelling and "no" if there is none."""


FUNCTION_REGENERATION = [{"role": "user", "content": f"""Instructions: \"The proposed function {{function_str}} if not working on the example. Can your provide another function string? Please return only the python function and nothing else. Include any imports. Do not provide any comments.\""""}]

REGEX_GENERATOR = [{"role": "system", "content": """You are an expert Regex user. I will give you a string transformation task. \
                         The task involves converting input strings to output strings. \
                         You will have to find the Regex pattern called "string_transformation_regex" that will generate the output string given the input string. \
                          Please return only the python function and nothing else. Include any imports. Do not provide any comments."""},\
                        {"role": "user", "content": f"""Instructions: {{instruction:}}"""}]



class PromptTemplate:
    def __init__(self, prompt) -> None:
        self.prompt = prompt

    def execute(self):
        pass

    def reset(self):
        self.inital_prompt = ""
        self.message_history = []

    def assign_new_prompt(self, new_prompt: List) -> None:
        self.prompt = new_prompt

    def add_prompt_incremental(self, prompt_to_add: Dict) -> None:
        self.prompt.append(prompt_to_add) 





