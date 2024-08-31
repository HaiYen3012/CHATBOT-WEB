from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from prompt_templates import memory_prompt_template
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize with Google API key and model name
GOOGLE_API_KEY = 'GOOGLE_API_KEY'
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm = llm, prompt = chat_prompt, memory = memory)

def load_normal_chain(chat_history):
    return ChatChain(chat_history)

class ChatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input=user_input, history = self.memory.chat_memory.messages,stop = ["Human: "])
    
