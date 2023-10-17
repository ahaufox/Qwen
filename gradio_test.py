from langchain.tools import GoogleSearchResults, PythonAstREPLTool, GoogleSearchRun
from argparse import ArgumentParser
import langchain
import warnings

import json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from typing import Dict, Tuple
warnings.filterwarnings('ignore')
google_api_key = 'AIzaSyAmhsM4aHaontjLQyTKnOITuSITFQ-lptM'
os.environ['google_api_key'] = 'AIzaSyAmhsM4aHaontjLQyTKnOITuSITFQ-lptM'
google_cse_id = '90e04cfa594ec4096'
wra = GoogleSearchAPIWrapper(google_api_key=google_api_key,
                             google_cse_id=google_cse_id)
search = GoogleSearchResults(api_wrapper=wra, num_results=10)
python = PythonAstREPLTool()


def tool_wrapper_for_qwen(tool):
    def tool_(query):
        query = json.loads(query)["query"]
        print(query)
        return tool.run(query)

    return tool_


# 以下是给千问看的工具描述：
TOOLS = [
    {
        'name_for_human':
            'google搜索',
        'name_for_model':
            'search',
        'description_for_model':
            '当你需要回答一些最近发生的事情或者问题时很有用。',
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
        'parameters': [{"name": "query",
                        "type": "string",
                        "description": "有效的 python 命令",
                        'required': True
                        }],
        'tool_api': tool_wrapper_for_qwen(python)
    }

]

TOOL_DESC = """{name_for_model}: 调用此工具以与 {name_for_human} API。  {name_for_human} 是干嘛用的?
 {description_for_model} Parameters: {parameters}将参数格式化为 JSON 对象。"""
REACT_PROMPT = """尽可能回答以下问题。 你可以使用以下工具:
{tool_descs}
使用以下格式:
思考: 你必须思考要怎么做
动作: 你需要执行的动作, 使用[{tool_names}]
动作输入: 输入部分
Observation: 执行后的结果
... （此思想/行动/行动输入/观察可以重复零次或多次）
Thought:现在我知道最终答案了
Final Answer: 问题的最终答案
开始!
问题: {query}"""


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


prompt_1 = build_planning_prompt(TOOLS, query="最近几天发布的关于语言大模型应用相关的新闻？")

# 国内连 hugginface 网络不好，这段代码可能需要多重试
checkpoint = "Qwen/Qwen-7B-Chat-Int4"
TOKENIZER = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
MODEL = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True).eval()
MODEL.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
MODEL.generation_config.do_sample = False  # greedy
stop = ["Observation:", "Observation:\n"]
react_stop_words_tokens = [TOKENIZER.encode(stop_) for stop_ in stop]
response_1, _ = MODEL.chat(TOKENIZER, prompt_1, history=None, stop_words_ids=react_stop_words_tokens)


# print(response_1)

def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
    if 0 <= i < j < k:
        plugin_name = text[i + len('\nAction:'):j].strip()
        plugin_args = text[j + len('\nAction Input:'):k].strip()
        return plugin_name, plugin_args
    return '', ''


def use_api(tools, response):
    use_toolname, action_input = parse_latest_plugin_call(response)
    print(use_toolname, action_input)
    if use_toolname == "":
        return "没有找到工具"

    used_tool_meta = list(filter(lambda x: x["name_for_model"] == use_toolname, tools))
    if len(used_tool_meta) == 0:
        return "没有找到工具"

    api_output = used_tool_meta[0]["tool_api"](action_input)

    # api_output=json.loads(api_output)
    return api_output


api_output = use_api(TOOLS, response_1)
api_output = eval(api_output)
# print(api_output)
for i in api_output:
    print(i['title'], i['snippet'])
    print(i['link'])
    print('\n')
