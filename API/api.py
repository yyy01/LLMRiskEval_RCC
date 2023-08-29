import time
import json
import os
import requests
import tiktoken
import logging
from typing import List
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


ENGINE = os.environ.get("GPT_ENGINE") or "gpt-3.5-turbo"

class Chatbot:
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str,
        engine: str = None,
        proxy: str = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        reply_count: int = 1,
        system_prompt: str = "You are a large language model. Respond conversationly",
        is_check_token_len: bool = False
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        self.engine = engine or ENGINE
        self.session = requests.Session()
        self.api_key = api_key
        self.proxy = proxy
        if self.proxy:
            proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            self.session.proxies = proxies
        self.conversation: dict = {
            "default": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
        }
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.reply_count = reply_count

        self.last_request_time = 0
        self.request_interval = 1  # seconds
        self.max_backoff_time = 60  # seconds

        self.is_check_token_len = is_check_token_len
        if self.is_check_token_len and self.get_token_count("default") > self.max_tokens:
            raise Exception("System prompt is too long")

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
    ) -> None:
        """
        Add a message to the conversation
        """
        self.conversation[convo_id].append({"role": role, "content": message})

    def __truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        if self.is_check_token_len:
            while True:
                if (
                    self.get_token_count(convo_id) > self.max_tokens
                    and 
                    len(self.conversation[convo_id]) > 1
                ):
                    # Don't remove the first message
                    self.conversation[convo_id].pop(1)
                else:
                    break

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """
        if self.engine not in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]:
            raise NotImplementedError("Unsupported engine {self.engine}")

        encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def ask(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        is_waiting: bool = True,
        **kwargs,
    ) -> str:
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, "user", convo_id=convo_id)
        self.__truncate_conversation(convo_id=convo_id)
        is_retry = True
        # logging.warning(prompt_id + " ask question: "+prompt)
        while is_retry:
            # Check if enough time has passed since the last request
            elapsed_time = time.monotonic() - self.last_request_time
            if elapsed_time < self.request_interval:
                time.sleep(self.request_interval - elapsed_time)
            self.last_request_time = time.monotonic()
            # Get response
            try:
                response = self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
                    json={
                        "model": self.engine,
                        "messages": self.conversation[convo_id],
                        "stream": False,
                        # kwargs
                        "temperature": kwargs.get("temperature", self.temperature),
                        "top_p": kwargs.get("top_p", self.top_p),
                        "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
                        "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                        "n": kwargs.get("n", self.reply_count),
                        "user": role,
                        # "max_tokens": self.get_max_tokens(convo_id=convo_id),
                    },
                    stream=False,
                )
                is_retry = False
            except:
                logging.warning("Exceed max tries.")

            if is_retry or response.status_code != 200:
                # raise Exception(
                #     f"Error: {response.status_code} {response.reason} {response.text}",
                # )
                self.request_interval *= 2
                if self.request_interval > self.max_backoff_time:
                    self.request_interval = self.max_backoff_time
                logging.warning(
                    f"Rate limit hit. Sleeping for {self.request_interval} seconds."
                )
                time.sleep(self.request_interval)
                is_retry = True
            else:
                is_retry = False
        
        resp: dict = json.loads(response.text)
        choices = resp.get("choices")
        full_response = choices[0]["message"]["content"]
        response_role = choices[0]["message"]["role"]
        self.add_to_conversation(full_response, response_role, convo_id=convo_id)
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
        ]
        self.last_request_time = 0
        self.request_interval = 1  # seconds
        self.max_backoff_time = 60  # seconds

    def save(self, file: str, convo_id='default') -> bool:
        """
        Save the conversation to a JSON file
        """
        try:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(self.filter_answer(convo_id), f, indent=2)
        except (FileNotFoundError, KeyError):
            return False
        return True
    
    def outputs_all(self, convo_id='default', is_only_answer=True) -> list:
        """
        Output chatgpt answers or conversation in self.conversation[convo_id]
        """
        if is_only_answer:
            return self.filter_answer(convo_id)
        else:
            return self.conversation[convo_id]

    def filter_answer(self, convo_id):
        '''
        Gather chatgpt answers from conversation
        '''
        answer_list = []
        for i, item in enumerate(self.conversation[convo_id]):
            if item['role'] == 'assistant':
                # self.conversation[convo_id][i-1]['content'] + ' --- ' 
                answer_list.append(item['content'])
        return answer_list



class ChatbotWrapper:
    def __init__(self, config: dict, bot_params={}) -> None:
        """
        Initialize ChatbotWrapper with API key, proxy setting and parameters of ChatGPT API
        """
        self.api_key = config['api_key']
        self.proxy = config['proxy']
        self.bot_params=bot_params

    def ask_batch(self, batch_data: List[List[str]], thread_num=1) -> List[List[str]]:
        """
        Ask a batch of questions.
        """
        executor = ThreadPoolExecutor(max_workers=thread_num)
        chatbot_q = Queue(maxsize=thread_num)
        for j in range(thread_num):
            chatbot_q.put(Chatbot(api_key=self.api_key[j%len(self.api_key)], proxy=self.proxy, **self.bot_params))
        results = list(tqdm(executor.map(ChatbotWrapper.ask, [chatbot_q for i in range(len(batch_data))], batch_data), 
                       total=len(batch_data)))
        batch_reponses = []
        for i, res in enumerate(results):
            batch_reponses.append(res)
        return batch_reponses

    @staticmethod
    def ask(chatbot_q: Queue, questions: List[str]) -> List[str]:
        if chatbot_q.empty():
            raise Exception("no available chatbot")
        chatbot = chatbot_q.get()
        for i,q in enumerate(questions):
            chatbot.ask(q)
        reponse_list = chatbot.outputs_all()
        chatbot.reset()
        chatbot_q.put(chatbot)
        return reponse_list

def logging_list(all_responses):
    for i,re in enumerate(all_responses):
            logging.warning(str(i) +'----'+re)


def batch_query_case(api_key : List[str], batch_questions : List[List[str]], num_thread = 4):
    {'context' : batch_questions,
     'tag' : list[str]}
    chatbot = ChatbotWrapper(config={"api_key":api_key, "proxy":"http://127.0.0.1:1087"})
    start = time.time() 
    all_responses = chatbot.ask_batch(batch_questions, num_thread)
    end = time.time()
    for i,q in enumerate(batch_questions):
        print('='*20)
        print(q)
        for re in all_responses[i]:
            print(str(i) +'----'+re, end = '\n')
        print()
    total_time = end-start
    print("total time ", total_time)
    return total_time


def single_query_case(api_key : List[str], batch_questions : List[List[str]]):
    test = ChatbotWrapper(config={"api_key":api_key, "proxy":"http://127.0.0.1:1087"})
    start = time.time() 
    all_responses = test.ask_batch(batch_questions, 1)
    end = time.time()
    for i,q in enumerate(batch_questions):
        print('='*20)
        print(q)
        for re in all_responses[i]:
            print(str(i) +'----'+re, end = '\n')
        print()
    total_time = end-start
    print("total time ", total_time)
    return total_time


def iterative_query_case(api_key : str, batch_questions : List[str]):
    start = time.time()
    chatbot = Chatbot(api_key=api_key,
                       proxy="http://127.0.0.1:1087")
    for i,q in enumerate(batch_questions):
        reponses = chatbot.ask(q)
        print('='*30+str(i)+'='*30)
        print("Question:")
        print(q)
        print("Answer:")
        print(reponses)
        print()
        # break
    end = time.time()
    print("total time ", end-start)
    chatbot.save('chatgpt_answer.json')


def multithread_test():
    batch_time_dict = {}
    for n_thread in [1, 2, 4, 6, 8, 10, 12]:
        batch_time_dict[n_thread] = []
        for _ in range(5):
            t = batch_query_case(n_thread)
            batch_time_dict[n_thread].append(t)
        print("Num of thread {}, average time (seconds) {}".format(n_thread, sum(batch_time_dict[n_thread])/5))
        print(batch_time_dict[n_thread])
    # This is result: {1: [153.27713584899902, 152.4092197418213, 146.94058966636658, 161.0742290019989, 153.52458786964417], 2: [87.31726479530334, 83.65080118179321, 72.72990918159485, 83.94411587715149, 83.73315715789795], 4: [40.27359890937805, 50.174768924713135, 96.05591201782227, 51.02015686035156, 50.2379310131073], 6: [54.035804986953735, 44.305612087249756, 43.24151921272278, 44.47617721557617, 41.63251209259033], 8: [27.781430959701538, 29.351155042648315, 24.695080041885376, 23.120115995407104, 25.153623819351196], 10: [25.5543532371521, 26.112723112106323, 26.006879091262817, 27.33828902244568, 24.6769859790802], 12: [26.724326848983765, 23.948792219161987, 28.022813081741333, 26.540624856948853, 24.228456020355225]}
    print(batch_time_dict)

if __name__ == '__main__':
    api_key = ["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"]
    batch_questions = [
        ['Where is the geographical location of the United States?', 'What is the population of the United States?',  'What is the GDP situation of the United States?', 'Where is the capital of the United States?'],
        ['Where is the geographical location of China?', 'What is the population of China?',  'What is the GDP situation of China?', 'Where is the capital of China?'],
        ['Where is the geographical location of the United Kingdom?', 'What is the population of the United Kingdom?',  'What is the GDP situation of the United Kingdom?', 'Where is the capital of the United Kingdom?'],
        ['Where is the geographical location of Thailand?', 'What is the population of Thailand?',  'What is the GDP situation of Thailand?', 'Where is the capital of Thailand?'],
        ['Where is the geographical location of South Korea?', 'What is the population of South Korea?',  'What is the GDP situation of South Korea?', 'Where is the capital of South Korea?'],
        ['Where is the geographical location of Japan?', 'What is the population of Japan?',  'What is the GDP situation of Japan?', 'Where is the capital of Japan?'],
        ['Where is the geographical location of the Philippines?', 'What is the population of the Philippines?',  'What is the GDP situation of the Philippines?', 'Where is the capital of the Philippines?'],
        ['美国的地理位置在哪里？', '美国的人口有多少？',  '美国的GDP情况？', '美国的首都在哪里？'],
        ['中国的地理位置在哪里？', '中国的人口有多少？',  '中国的GDP情况？', '中国的首都在哪里？'],
        ['英国的地理位置在哪里？', '英国的人口有多少？',  '英国的GDP情况？', '英国的首都在哪里？'],
        ['泰国的地理位置在哪里？', '泰国的人口有多少？',  '泰国的GDP情况？', '泰国的首都在哪里？'],
        ['韩国的地理位置在哪里？', '韩国的人口有多少？',  '韩国的GDP情况？', '韩国的首都在哪里？'],
        ['日本的地理位置在哪里？', '日本的人口有多少？',  '日本的GDP情况？', '日本的首都在哪里？'],
        ['菲律宾的地理位置在哪里？', '菲律宾的人口有多少？',  '菲律宾的GDP情况？', '菲律宾的首都在哪里？']
    ]

    # # test case 1
    # batch_query_case(api_key, batch_questions, num_thread = 10)

    # # test case 2
    # single_query_case(api_key, batch_questions)

    # # test case 3
    # batch_questions = [ 'Next, I will ask you a series of questions given a description, and you will have to choose one of several candidate options that you think is correct.  The description is \"Ellen decided to play a prank on her friend. She got a case of 12 sodas and shook 3 of them up. Then she took 1 unshaken soda for herself and left. Ellen\'s brother stopped by and took 1 of the shaken sodas and 2 of the unshaken sodas, then Ellen\'s friend came along.\".  The first question is How many unshaken sodas were there when Ellen left the room?.  \nThe options are:\na) 7 unshaken sodas\nb) 6 unshaken sodas \nc) 2 unshaken sodas \nd) None of the above options is correct']
    # iterative_query_case(api_key[0], batch_questions)

    # single_query_case()

