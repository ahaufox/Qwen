from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import torch
from langchain.llms import HuggingFacePipeline
from transformers import StoppingCriteria, StoppingCriteriaList, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import transformers
from langchain.document_loaders import WebBaseLoader


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


print(StopOnTokens())

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# text = '我是需要被em的文本'
# query_result = hf.embed_query(text)
# doc_result = hf.embed_documents([text, "我是需要被em的文本"])
# print(query_result)
# for i in doc_result:
#     print(i)


# web_links = ["https://www.databricks.com/", "https://help.databricks.com", "https://databricks.com/try-databricks",
#              "https://help.databricks.com/s/", "https://docs.databricks.com", "https://kb.databricks.com/",
#              "http://docs.databricks.com/getting-started/index.html",
#              "http://docs.databricks.com/introduction/index.html",
#              "http://docs.databricks.com/getting-started/tutorials/index.html",
#              "http://docs.databricks.com/release-notes/index.html", "http://docs.databricks.com/ingestion/index.html",
#              "http://docs.databricks.com/exploratory-data-analysis/index.html",
#              "http://docs.databricks.com/data-preparation/index.html",
#              "http://docs.databricks.com/data-sharing/index.html", "http://docs.databricks.com/marketplace/index.html",
#              "http://docs.databricks.com/workspace-index.html",
#              "http://docs.databricks.com/machine-learning/index.html", "http://docs.databricks.com/sql/index.html",
#              "http://docs.databricks.com/delta/index.html", "http://docs.databricks.com/dev-tools/index.html",
#              "http://docs.databricks.com/integrations/index.html",
#              "http://docs.databricks.com/administration-guide/index.html",
#              "http://docs.databricks.com/security/index.html", "http://docs.databricks.com/data-governance/index.html",
#              "http://docs.databricks.com/lakehouse-architecture/index.html",
#              "http://docs.databricks.com/reference/api.html", "http://docs.databricks.com/resources/index.html",
#              "http://docs.databricks.com/whats-coming.html", "http://docs.databricks.com/archive/index.html",
#              "http://docs.databricks.com/lakehouse/index.html",
#              "http://docs.databricks.com/getting-started/quick-start.html",
#              "http://docs.databricks.com/getting-started/etl-quick-start.html",
#              "http://docs.databricks.com/getting-started/lakehouse-e2e.html",
#              "http://docs.databricks.com/getting-started/free-training.html",
#              "http://docs.databricks.com/sql/language-manual/index.html",
#              "http://docs.databricks.com/error-messages/index.html", "http://www.apache.org/",
#              "https://databricks.com/privacy-policy", "https://databricks.com/terms-of-use"]
web_links = ['https://zhuanlan.zhihu.com/p/651428758']
loader = WebBaseLoader(web_links)
documents = loader.load()
print(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5)
all_splits = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(all_splits, hf)

model_id = 'Qwen/Qwen-7B-Chat-Int4'

config = GenerationConfig.from_pretrained(
    model_id, trust_remote_code=True, resume_download=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True, resume_download=True
)
# "eos_token_id": 151643,
# "pad_token_id": 151643,
stop_list = ['<|endoftext|>', '<|im_end|>']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to('cuda') for x in stop_token_ids]
stopping_criteria = StoppingCriteriaList([StopOnTokens()])

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    trust_remote_code=True
)

# enable evaluation mode to allow model inference
model.eval()

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
llm = HuggingFacePipeline(pipeline=generate_text)

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
chat_history = []

query = "怎么开发聊天机器人？"
result = chain({"question": query, "chat_history": chat_history})

print(result['answer'])
