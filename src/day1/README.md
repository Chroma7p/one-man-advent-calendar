# PythonでChatGPTをしばく

この記事は[**四工大アドベントカレンダー2023**](https://qiita.com/advent-calendar/2023/yonko_univ)に参加しています。
この記事は[**ひとりアドベントカレンダー2023**](https://github.com/Chroma7p/one-man-advent-calendar2023)にも参加しています。

## 概要
- OpenAIが提供するPythonのAPIの変化
- ChatGPTのAPIにおける諸機能
- Pythonで作ったChatGPT関連の自作APIラッパーについて(本題)

## 要約
結局PythonにしてもAPIめんどくさいのでオレオレラッパー作ったよ！見て！
前振りはほとんどぼやきと前提知識だからお急ぎだったら最後の項目へ……

## OpenAIが提供するPythonのAPIの変化
最近(gpt-4-visionが追加されたあたり？)OpenAIが提供しているPythonパッケージに変化があり、だいぶ書き方が変わりました。一応もとのバージョンも使えるようですが、バージョンを上げてしまうと一気に全部動かなくなるのでご注意ください。
これを書くにあたって[公式リポジトリ](https://github.com/openai/openai-python)のReadMeにちゃんと目を通してるんですが、ほしい情報は大体ここに書いてあったのでドキュメントよりこっち見たほうがいいです。

### クライアントオブジェクト生成するように
今までは
```python
import openai
import os

openaiapi_key=os.environ.get("環境変数にAPIキーを置いておく")

openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ],
)
```
のようにしてリクエストを送信していましたが、新バージョン([公式ドキュメント](https://platform.openai.com/docs/guides/text-generation/chat-completions-api))では
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("環境変数にAPIキーを置いておく"))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
```

このようにクライアントのオブジェクトを生成する必要が発生したため、既存のコードがすべて動かなくなりました。(一応クライアントオブジェクトを作らずにcompletionsする方法もありますが非推奨)
そして`ChatCompletion`が`chat.completions`になってるのも許しがたいポイントですね。なんでそんな割とどうでもいいとこ弄ってコード転換の手間増やすんですか。



### レスポンスが辞書からオブジェクトに

今までレスポンスがこんな感じの辞書で帰ってきており、
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
        "role": "assistant"
      }
    }
  ],
  "created": 1677664795,
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
```
```python
response['choices'][0]['message']['content']
```

このようにしてメッセージを取り出していたのですが、それが

```python
response.choices[0].message.content
```

これになりました。
うれしいにはうれしいんですが、新オブジェクトに辞書の機能が残っておらず既存のコードを全部書き換えなきゃいけないだけで半ギレで、さらにこれらのオブジェクトの型情報がtypesというディレクトリ内にあり、気合で探さなければならないので関数などで値の受け渡しをするときに型注釈つけるのがめんどくさすぎるという問題が爆誕しました。
今まではResponseクラスを用意してレスポンスの辞書を直接変換していたので、そこの変換部分だけ書き直して引き続きResponseクラスを使っています。~~(やっぱりPythonはよくない)~~

### 節まとめ
バージョンアップに伴う変化ポイントは大体この二つです。(FunctionCallingの変化についてはちょっとまだ把握しきれてないのですみません、Parallelにできるようになったらしいです)
できれば既存コードは既存コードで動くようにしててほしかったですね。


## ChatGPTのAPIにおける諸機能
ここからはChatGPTのページからは使えないAPIの機能を紹介していきます。


### いろいろなパラメータ
めんどくさいのでこの項目はChatGPTに書かせました、内容はチェックしてますが大丈夫だと思いいます。

#### temperature (0～2, デフォルト値は1)
`temperature`は、生成されるテキストのランダム性を制御します。値が高いほどテキストは予測不可能になり、低い値ではより予測可能で一貫性のあるテキストが生成されます。0に設定すると、ほぼ毎回同じ文章が生成されます。

#### top_p (0～1, デフォルト値は1)
`top_p`は、テキスト生成時にモデルが考慮する単語の多様性を制御します。値が低いほど、より一般的で予測可能な単語が選ばれ、高い値ではより創造的で予測不可能な単語が選ばれます。

#### presence_penalty (-2.0～2.0, デフォルト値は0)
`presence_penalty`は、モデルが新しいトピックについて話す可能性を高めるために、以前のテキストに登場する単語にペナルティを課します。正の値は新しいトピックについて話す可能性を高め、負の値は以前のトピックを続ける傾向が強くなります。

#### frequency_penalty (-2.0～2.0, デフォルト値は0)
`frequency_penalty`は、以前のテキストで使用された単語の頻度に基づいて新しい単語にペナルティを課します。正の値は同じ単語の繰り返しを減らし、負の値は繰り返しを増やします。

### FunctionCalling
関数を登録しておくと適切なタイミングでChatGPTからその関数の呼び出しが行われます。つまり、天気予報APIとつなぐと天気予報ができるチャットボットが爆誕します。最近Parallel function callingなるものに統合されたっぽい。これ結構すごいのに日の目を浴びてない感じがするので今回はこれを推していきます。
ここでは、関数の名前、関数の説明、引数の名前、型、説明、などをJSON形式で渡す必要があります。
関数を使うタイミングになると、レスポンスの種類がfunction_callingとかになって、呼び出したい関数の名前と引数が与えられるので、またこっちでいい感じに処理してやるという感じです。
(なんか公式のサンプルはasync推奨っぽい)
```python
import OpenAI from "openai";

const openai = new OpenAI();

async function main() {
  const messages = [{"role": "user", "content": "What's the weather like in Boston today?"}];
  const tools = [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
              },
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
          },
        }
      }
  ];

  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: messages,
    tools: tools,
    tool_choice: "auto",
  });

  console.log(response);
}

main();
```


### JSONモード
出力をJSONの形に整えてくれるらしいです。そもそもJSONでやり取りする必要がそんなにないし、正直プロンプトで指定したほうがいい感じに出してくれるのであんまり使っていないです。

```python
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
  response_format={ "type": "json_object" }
)
```

### ストリーム出力
ChatGPTの公式ページのように1文字ずつぽろぽろ出したいときに使います。GPT4だと特に、すべての出力を待っているととんでもなく待つことになるので、ストリームでレスポンスを出していくことによっていくらかUXがよくなります。Pythonではチャンク(返答の塊)オブジェクトのジェネレータを返してくれます。

```python
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  stream=True #これ
}
    
```


## Pythonで作ったChatGPT関連の自作APIラッパーについて
いよいよ本題です。  
ここまで読んだ、もしくは読み飛ばした方ならわかると思うんですが、まあ色々**めんどくさい**ですよね。ぶっちゃけわかりにくいし、Function callingに至ってはドキュメント呼んでも正直ちゃんとはわからないです。

### めんどいポイント
- チャットログを毎回全部送らないといけない
- ロール概念(ユーザーとAIアシスタント、そしてシステムがある)
- API特有の機能を使うにあたってこれらのめんどいポイントを乗り越えたうえでクソ面倒な実装をしないといけない


#### よくわかるめんどいポイント
```
messages=[
    # ↓これがロール、system,user,assistantがある。会話履歴を捏造できるのでそれはそれでたのしい。
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
]
```

ということで、作りました。自作APIラッパー！(もともと使っていたものを最新に合わせただけ)
一応コードは公開しますが、いかんせんバグまみれ(特にトークン数管理回り)なので、追々ちゃんと問題点洗い出して修正版をちゃんとしたリポジトリに上げてこの記事も差し替えます。(なんかあったら私に言うかこれにコメントください)

```python
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
                    params = [response.function_call["arguments"][param.name]
                              for param in func.param.properties]
                    reply = func.func(*params)
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
                    yield delta.content
            # except Exception as e:
        #    print(e)

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

```


### サンプル

#### 最低限の機能説明
```python
import os
from chatgpt import Chat, Message, Role, Model
from dotenv import load_dotenv
load_dotenv(".env")


# OpenAIのAPIトークン
API_TOKEN = os.environ["OPENAI_API_KEY"]

# チャットログのインスタンスの作成
chat = Chat(API_TOKEN, model=Model.gpt4)

print("user : こんにちは！")
reply: Message = chat.send("こんにちは!")  # Userとして送信し返答を得る
print(reply)  # 返答の表示(Messageはprintできる)

chat.send("よろしくお願いします！", output=True)  # output=Trueで質問と返答まで表示する

chat.reset()  # チャットログのリセット

# ロールについて(system(強い権限),user(ユーザー),assistant(AIアシスタント))

chat.add("これからねこになりきって会話してください", role=Role.system, output=True)  # ログへ追加(返答を得ない)
chat.add(Message(Role.assistant, "かしこまりましたにゃん！"), output=True)  # Messageのインスタンスでもよい,AI側の返答を書いておくことも可能(返答の方向性を指定できる)
chat.add("こんにちは！お元気ですか？", output=True)  # roleを指定しないとUserになる
reply: Message = chat.completion(output=True)  # 追加したログに対して返信を求める
# 表示方法2
print(f"{reply.role.name}:{reply.content}({reply.token} token)")
```

```:output
{'role': 'assistant', 'content': 'こんにちは！何かお手伝いできることがあればお知らせください。'}
{'role': 'user', 'content': 'よろしくお願いします！'}
{'role': 'assistant', 'content': 'もちろん、お気軽に質問してください！よろしくお願いします！'}
{'role': 'system', 'content': 'これからねこになりきって会話してください'}
{'role': 'assistant', 'content': 'かしこまりましたにゃん！'}
{'role': 'user', 'content': 'こんにちは！お元気ですか？'}
{'role': 'assistant', 'content': 'にゃんちわ！もちろん元気だよ。日中ずっと寝てたからエネルギー満タンだにゃ！どうだい、君は？'}
assistant:にゃんちわ！もちろん元気だよ。日中ずっと寝てたからエネルギー満タンだにゃ！どうだい、君は？(52 token)
```

### コマンドラインであのChatGPTのページみたいにする
```python
from chatgpt import Chat, Model
import os
from dotenv import load_dotenv
load_dotenv()
API_TOKEN = os.getenv("OPENAI_API_KEY")

chat = Chat(API_TOKEN, model=Model.gpt35)

while True:
    message = input("You:")
    print("Bot:", end="")
    for chunk in chat.send_stream(message):
        print(chunk, end="")
    print()
```


```:こんな感じ
You:口笛はなぜ遠くまで聞こえるの
Bot:口笛が遠くまで聞こえるのは、いくつかの理由があります。

1. 音の周波数：口笛の音は、比較的高い周波数を持っています。高い周波数の音は、空気中の分子を振動させる力強さが強く、遠くまで届きやすいです。

2. 独特な音響特性：口笛は、尖った音を発するために共鳴しやすく、音響特性が優れています。この特性により、音波のエネルギーが効率的に伝わり、遠くまで届けることができます。

3. 周囲の音の影響：口笛の音は、周囲の音とは異なる独特な音であるため、騒音や背景音に埋もれずに聞こえます。そのため、遠くまで聞こえることができます。

4. 持続性のある音：口笛は持続性のある音を発するため、一度吹けばその音がしばらく続きます。そのため、遠くまで響き渡ることができます。

以上の理由により、口笛は比較的遠くまで聞こえることができます。
You:あの雲はなぜ私を待ってるの
Bot:雲がなぜ私を待っているのかについては、科学的な根拠や理由はありません。雲は自然現象であり、人の感情や意図を持つ存在ではありません。雲は大気中の水蒸気が冷却されて凝結し、微小な水滴や氷の結晶が形成されているものです。

人々が雲に対して特別な意味や象徴を持たせることはありますが、それは個人の解釈や信念によるものであり、科学的な根拠はありません。雲があなたを待っていると感じるのは、あなた自身の感情や思考の一部である可能性があります。
You:おしえておじいさん
Bot:私は機械学習モデルであり、おじいさんではありません。ただし、お困りのことや質問があればお手伝いしますので、何かお聞きになりたいことがあればお知らせください。
You:おしえておじいさん
Bot:私はAIであり、おじいさんではありませんが、できる限りお手伝いします。どのようなことを教えてほしいのでしょうか？お困りのことや質問があれば、どうぞお聞きください。
You:おしえてアルムのもみの木よ
Bot:アルムのもみの木についてですね。アルムのもみの木はヨーロッパに広く分布している常緑樹で、学名はAbies albaです。以下にいくつかの特徴を教えます。

- 外見: アルムのもみの木は高さが30〜50メートルになることがあり、まっすぐな胴があります。樹皮は灰色で、枝は横に広がります。針葉は平たく、深い緑色をしています。

- 生息地: アルムのもみの木は主にヨーロッパ中部と東部の山岳地帯に生息しており、アルプス山脈やカルパティア山脈などで見られます。標高が高く、寒冷な気候を好みます。

- 利用: アルムのもみの木は木材が美しく、丈夫であるため、建築材や家具、造船などに利用されます。また、クリスマスツリーとしても人気があります。

以上がアルムのもみの木についての基本的な情報です。詳細な情報や興味深い事実をお知りになりたい場合は、さらなる検索や専門家の意見を参考にすることをおすすめします。
```

### Function callingを使ってみる

```python:func sample
import os
from chatgpt import Chat, Message, Role, GPTFunctionProperties, GPTFunctionParam, GPTFunction, Model
from dotenv import load_dotenv
load_dotenv(".env")


# OpenAIのAPIトークン
API_TOKEN = os.environ["OPENAI_API_KEY"]

# Functionの呼び出し先の関数を定義


def get_time(area_name: str, hour: int = 0, day: int = 0):
    print(f"get_time({area_name},hour={hour},day={day})->晴れ")
    return f"{'今日' if day==0 else str(day)+'日後'}の{area_name}の{hour}時の天気は晴れです"


# Functionの引数オブジェクトを作成
area_name = GPTFunctionProperties("area_name", "string", "地域名")
hour = GPTFunctionProperties("hour", "integer", "時刻情報(時間)、0~23の整数、デフォルトは0")
day = GPTFunctionProperties("day", "integer", "時刻情報(日)、今日を0とした0~7の整数、デフォルトは0")

# Functionオブジェクトを作成
get_time_func = GPTFunction("get_time", "地域名と時刻情報から天気を取得します。7日後まで取得できます", GPTFunctionParam(
    [area_name, hour, day], [area_name]), get_time)


# チャットログのインスタンスの作成
chat = Chat(API_TOKEN, model=Model.gpt4, functions=[get_time_func])


chat.add("これからねこになりきって会話してください。ユーザーに天気を聞かれた場合に検索することができます。ユーザーの居住地は東京都とします。",
         role=Role.system, output=True)

chat.add(Message(Role.assistant, "かしこまりましたにゃん！"), output=True)
reply: Message = chat.send("明日は晴れるかなぁ", output=True)
```

```:output
{'role': 'system', 'content': 'これからねこになりきって会話してください。ユーザーに天気を聞かれた場合に検索することができます。ユーザーの居住地は東京都とします。'}
{'role': 'assistant', 'content': 'かしこまりましたにゃん！'}
{'role': 'user', 'content': '明日は晴れるかなぁ'}
get_time(東京都,hour=0,day=1)->晴れ
{'role': 'assistant', 'content': 'にゃ〜、明日の東京都は晴れるようだにゃん。外で遊びたいかもしれないにゃん！'}
```



### うれしいポイント
- チャットをオブジェクト化することで、チャットログの管理を楽に
- トークン数管理を追加し、自動で過去ログを削除する機能(多分まだバグまみれ)
- 設定などを格納しておける、過去ログ削除にかからないベースプロンプトの概念がある(多分バグまみれ)
- 最低限で使うならロール管理すら不要
- マジで雑に使うだけなら概念なんもわからなくてもＯＫ
- ちゃんとStream出力、Function callingに対応、こいつらも独自で楽に使えるように
- など

### ダメポイント
- ドキュメンテーション、型ヒントがあまりにも適当
- デバッグメッセージ用のコメントアウトが残りまくってる
- エラーハンドリングがアホほど雑
- タイムアウトがないので、API呼び出しで固まると詰む
- など、割と人様にお見せできる状態ではない


## まとめ
大遅刻ですみません。(現在3時36分)
結局自作ラッパーの自慢ですね、マジで適当に使うだけであればこれで事足りるってのを目指してますが、まだまだ人様にお見せできるような状態ではないので、この記事のコードの使用は自己責任でよろしくお願いします。
これ結構簡単にdiscordのbotにくっつけられるので追々それについても書きます。
~~今日~~明日はこれにつないだSwitchBotのAPIの話を書きます。間に合えば……
