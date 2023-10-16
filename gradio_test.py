from langchain.tools import GoogleSearchResults, PythonAstREPLTool, GoogleSearchRun
from argparse import ArgumentParser
import langchain
import json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from langchain.utilities.google_search import GoogleSearchAPIWrapper

google_api_key = 'AIzaSyAmhsM4aHaontjLQyTKnOITuSITFQ-lptM'
os.environ['google_api_key'] = 'AIzaSyAmhsM4aHaontjLQyTKnOITuSITFQ-lptM'

search = GoogleSearchAPIWrapper(google_api_key='AIzaSyAmhsM4aHaontjLQyTKnOITuSITFQ-lptM',
                                google_cse_id='qwen-tool')
python = PythonAstREPLTool()


def tool_wrapper_for_qwen(tool):
    def tool_(query):
        query = json.loads(query)["query"]
        return tool.run(query)

    return tool_


# 以下是给千问看的工具描述：
TOOLS = [
    {
        'name_for_human':
            'google search',
        'name_for_model':
            'Search',
        'description_for_model':
            '当你需要回答有关当前事件的和一些问题时很有用。',
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "在google上进行搜索",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(search)
    },
    {
        'name_for_human':
            'python',
        'name_for_model':
            'python',
        'description_for_model':
            "一个Python shell. 使用它来执行 python 命令。使用此工具时, 有时输出是缩写的 - 在答案中使用它之前，请确保它看起来没有缩写. "
            "不要在 python 代码中添加注释。",
        'parameters': [{
            "name": "query",
            "type": "string",
            "description": "有效的 python 命令",
            'required': True
        }],
        'tool_api': tool_wrapper_for_qwen(python)
    }

]

TOOL_DESC = """{name_for_model}: 调用此工具以与 {name_for_human} API. What is the {name_for_human} API useful for?
 {description_for_model} Parameters: {parameters}将参数格式化为 JSON 对象。"""
REACT_PROMPT = """尽可能回答以下问题。 你可以使用以下工具:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


def build_planning_prompt(TOOLS, query):
    tool_descs = []
    tool_names = []
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model=info['name_for_model'],
                name_for_human=info['name_for_human'],
                description_for_model=info['description_for_model'],
                parameters=json.dumps(
                    info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)

    prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
    return prompt


prompt_1 = build_planning_prompt(TOOLS[0:1], query="加拿大2023年人口统计数字是多少？")
print(prompt_1)
# 国内连 hugginface 网络不好，这段代码可能需要多重试
checkpoint = "Qwen/Qwen-7B-Chat-Int4"
TOKENIZER = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
MODEL = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True).eval()
MODEL.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
MODEL.generation_config.do_sample = False  # greedy
stop = ["Observation:", "Observation:\n"]
react_stop_words_tokens = [TOKENIZER.encode(stop_) for stop_ in stop]
response_1, _ = MODEL.chat(TOKENIZER, prompt_1, history=None, stop_words_ids=react_stop_words_tokens)
print(response_1)
