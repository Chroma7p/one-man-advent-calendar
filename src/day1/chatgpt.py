from enum import Enum
import openai
import tiktoken
from typing import Callable
import json
from openai.types import chat
from openai.types.chat.chat_completion_assistant_message_param import FunctionCall


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    function = "function"


class Model(Enum):
    gpt35 = "gpt-3.5-turbo"
    gpt4 = "gpt-4"
    gpt4vp = "gpt-4-vision-preview"
    gpt4p = "gpt-4-1106-preview"


class GPTFunctionProperties():
    def __init__(self, name: str, param_type: str, description: str, enum: Enum | None = None):
        self.name = name
        self.param_type = param_type
        self.description = description
        if enum is None:
            self.enum = []
        elif isinstance(enum, Enum):
            self.enum = enum
        elif issubclass(enum, Enum):
            self.enum = [e.value for e in enum]
        else:
            self.enum = enum

    def to_json(self):
        if self.enum:
            return {"type": self.param_type,  "enum": self.enum}
        return {"type": self.param_type, "description": self.description}


class GPTFunctionParam():
    def __init__(self, properties: list[GPTFunctionProperties], required: list[GPTFunctionProperties]):
        self.properties = properties
        self.required = required

    def to_json(self):
        return {"type": "object", "properties": {prop.name: prop.to_json() for prop in self.properties}, "required": [prop.name for prop in self.required]}


class GPTFunction():
    def __init__(self, name: str, description: str, param: GPTFunctionParam, func: Callable):
        self.name = name
        self.description = description
        self.param = param
        self.func = func

    def to_json(self):
        obj = {"name": self.name, "description": self.description}
        obj["parameters"] = self.param.to_json()
        return obj


class Message:
    """
    メッセージのクラス
    メッセージごとにロールと内容とトークンを保持する
    """

    def __init__(self, role: Role, content: str, token: int = 0, name: str = "", img: list[str] = []):
        self.role: Role = role
        self.content: str = content
        if not self.content:
            self.content = ""
        self.token: int = token
        if token == 0:
            self.calc_token()
        self.name: str = name
        self.img: list[str] = img

    def msg2dict(self) -> dict:
        if self.role == Role.function:
            return {"role": self.role.name, "content": self.content, "name": self.name}
        if len(self.img) > 0:
            return {"role": self.role.name, "content": [{"type": "text", "text": self.content}]+[{"type": "image_url", "image_url": self.img[i]} for i in range(len(self.img))]}
        return {"role": self.role.name, "content": self.content}

    def set_token(self, token: int) -> None:
        self.token = token

    def msg2str(self) -> str:
        return f"{self.role.name} : {self.content}"

    def __str__(self) -> str:
        return str(self.msg2dict())
        return self.msg2str()

    def calc_token(self):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if self.content:
            self.token = len(encoding.encode(self.content))
        else:
            self.token = 0


class Response:
    """
    レスポンスのクラス
    必要な情報を雑にまとめる
    """

    def __init__(self, response: chat.chat_completion.ChatCompletion):
        self.choices: dict = response.choices
        if self.choices:
            self.messages: list[Message] = [Message(Role(
                choice.message.role), choice.message.content)for choice in self.choices]
        self.created: int | None = response.created
        self.id: str | None = response.id
        self.model: str | None = response.model
        self.completeion_tokens: int | None = response.usage.completion_tokens
        self.prompt_tokens: int | None = response.usage.prompt_tokens
        self.function_call: FunctionCall | None = self.choices[0].message.function_call
        if self.function_call:
            try:
                self.function_call = {"name": self.function_call.name, "arguments": json.loads(
                    self.function_call.arguments)}
            except:
                print(self.function_call.arguments)
                exit()

        """
        print(self.choices)
        print(self.messages)
        print(self.created)
        print(self.id)
        print(self.model)
        print(self.completeion_tokens)
        print(self.prompt_tokens)
        """


class Chat:
    """
    チャットのクラス
    """

    def __init__(self, API_TOKEN: str, organization: str | None = None, model: Model = Model.gpt35, functions: list[GPTFunction] = [], base_prompts: list[Message] = [], TOKEN_LIMIT: int = 4096, n: int = 1, thin_out_flag: bool = False) -> None:
        self.organization: str | None = organization
        self.history: list[Message] = []
        self.model: Model = model
        self.TOKEN_LIMIT: int = TOKEN_LIMIT
        self.n: int = n
        self.thin_out_flag: bool = thin_out_flag
        self.API_TOKEN: str = API_TOKEN
        self.functions: list[GPTFunction] = functions
        self.base_prompts: list[Message] = base_prompts
        openai.api_key = self.API_TOKEN
        if self.organization:
            openai.organization = self.organization
        self.client = openai.OpenAI()

    def add(self, message: list[Message] | Message, role: Role = Role.user, output: bool = False) -> None:
        """
        トークログの末尾にメッセージを追加
        """

        if type(message) is str:
            message = Message(role, message)
            self.history.append(message)
            if output:
                print(message)
        elif type(message) is list:
            if output:
                for msg in message:
                    print(msg)
            self.history.extend(message)
        elif type(message) is Message:
            self.history.append(message)
            if output:
                print(message)
        else:
            raise Exception("can't add anything that is not a message")

    def completion(self, output: bool = False) -> Message:
        """
        現在の履歴の状態で返信を得る
        戻り値はMessaegeクラス
        """

        response = self.create()

        if response.function_call:
            for func in self.functions:
                if func.name == response.function_call["name"]:
                    params = {param.name: response.function_call["arguments"][param.name]
                              for param in func.param.properties if param.name in response.function_call["arguments"]}
                    reply = func.func(**params)
                    reply = Message(Role.function, reply, name=func.name)
                    self.add(reply)
                    return self.completion(output=output)
            else:
                print("no function")
                exit()
        completion_token = response.completeion_tokens
        reply: Message = response.messages[0]
        reply.set_token(completion_token)

        completion_token = response.completeion_tokens
        reply: Message = response.messages[0]
        reply.set_token(completion_token)
        self.add(reply)
        if output:
            print(reply)
        return reply

    def send(self, message: str | Message, role: Role = Role.user, output: bool = False, stream: bool = False) -> Message:
        """
        メッセージを追加して送信して返信を得る
        messageがMessageクラスならそのまま、strならMessageクラスに変換して送信
        add+completionみたいな感じ
        戻り値はMessageクラス
        """
        if type(message) is str:
            message = Message(role, message)

        if self.get_now_token() + len(message.content) > self.TOKEN_LIMIT:
            # トークン超過しそうなら良い感じに間引くかエラーを吐く
            if self.thin_out_flag:
                self.thin_out()
            else:
                raise Exception("token overflow")

        self.add(message, output=output)
        reply = self.completion(output=output)
        self.history.append(reply)
        return reply

    def send_stream(self, message: str | Message, role: Role = Role.user, output: bool = False):
        if type(message) is str:
            message = Message(role, message)
        if message.content:
            if self.get_now_token() + len(message.content) > self.TOKEN_LIMIT:
                # トークン超過しそうなら良い感じに間引くかエラーを吐く
                if self.thin_out_flag:
                    self.thin_out()
                else:
                    raise Exception("token overflow")
        self.add(message, output=output)
        openai.api_key = self.API_TOKEN
        log = self.make_log()
        if self.organization:
            openai.organization = self.organization
        if self.functions:
            response: list[chat.ChatCompletionChunk] = self.client.chat.completions.create(
                model=self.model.value,
                messages=log,
                max_tokens=1000,
                functions=[func.to_json()
                           for func in self.functions] if self.functions else None,
                n=self.n,
                stream=True,
            )
        else:
            response: list[chat.ChatCompletionChunk] = self.client.chat.completions.create(
                model=self.model.value,
                messages=log,
                n=self.n,
                stream=True,
            )
        function_call = None
        content = ""
        for chunk in response:
            if type(chunk) is not chat.ChatCompletionChunk:
                continue
            delta = chunk.choices[0].delta
            # print(chunk)
            if delta.function_call or function_call:
                if not function_call:
                    function_call = dict(delta.function_call)
                else:
                    if delta.function_call:
                        if delta.function_call.name:
                            function_call["name"] += delta.function_call.name
                        if delta.function_call.arguments:
                            function_call["arguments"] += delta.function_call.arguments
                # print(function_call)
                if not chunk.choices[0].finish_reason:
                    continue
                # print(function_call)
                if function_call["arguments"] != "":
                    function_call["arguments"] = json.loads(
                        function_call["arguments"])
                for func in self.functions:
                    if func.name == function_call["name"]:
                        # print(f"func:{func.name} args:{function_call['arguments']}")
                        params = [function_call["arguments"][param.name]
                                  for param in func.param.properties]
                        reply = func.func(*params)
                        # print(f"func:{func.name} reply:{reply}")
                        reply = Message(Role.function, reply, name=func.name)
                        for i in self.send_stream(reply):
                            yield i
                        break
                else:
                    for i in self.send_stream(Message(Role.system, "Function not found")):
                        yield i

            else:
                if delta.content:
                    content += delta.content
                    yield delta.content
            # except Exception as e:
        #    print(e)
        self.add(Message(Role.user, content))

    def make_log(self) -> list[dict]:
        """
        メッセージインスタンスのリストをAPIに送信する形式に変換
        """
        return [hist.msg2dict() for hist in self.base_prompts+self.history]

    def get_now_token(self) -> int:
        """
        現在のトークン数を取得
        """
        return sum([x.token for x in self.base_prompts+self.history])

    def thin_out(self, n: int | None = None) -> None:
        """
        トークログをTOKEN_LIMITに基づいて8割残すように先頭から消す
        引数nで減らす分のトークン数を指定
        """
        if not n:
            limit = self.TOKEN_LIMIT * 0.8
        else:
            limit = self.TOKEN_LIMIT - n
        now_token = self.get_now_token()
        remove_token = 0
        remove_index = 0
        while now_token - remove_token > limit:
            remove_token += self.history[remove_index].token
            remove_index += 1
        self.history = self.history[remove_index:]

    def create(self) -> Response:
        """
        openaiのAPIを叩く
        """
        openai.api_key = self.API_TOKEN
        if self.organization:
            openai.organization = self.organization
        log = self.make_log()
        # print(log)
        if self.functions:
            response = self.client.chat.completions.create(
                model=self.model.value,
                messages=log,
                functions=[func.to_json()
                           for func in self.functions] if self.functions else None,
                n=self.n,
                max_tokens=1000,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model.value,
                messages=log,
                n=self.n,
                max_tokens=1000,
            )

        return Response(response)

    def get_history(self) -> str:
        """
        会話ログをテキスト化
        """
        text: str = ""

        for i, msg in enumerate(self.history):
            text += f"{i:03}:{msg.msg2str()[:-20]}\n"

        return text

    def remove(self, index: int) -> None:
        """
        ログの一部削除
        """
        if not 0 <= index < len(self.history):
            raise Exception("index out of range")
        self.history.remove(index)

    def reset(self):
        """
        ログの全削除
        """
        self.history = []
