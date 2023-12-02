# PythonでSwitchBotのAPIをしばく

この記事は[**四工大アドベントカレンダー2023**](https://qiita.com/advent-calendar/2023/yonko_univ)に参加しています。
この記事は[**ひとりアドベントカレンダー2023**](https://github.com/Chroma7p/one-man-advent-calendar2023)にも参加しています。

2日目です。

## 概要
- Pythonを使用したSwitchBot API v1.1でのデバイス制御
- より使いやすくする

## 要約
SwitchBot API v1.1を使用して、Pythonでデバイスを制御する方法を紹介します。
また、デバイスごとにクラスを作成して、より使いやすくする方法も紹介します。


## Pythonを使用したSwitchBot API v1.1でのデバイス制御
SwitchBotのデバイスをPythonで制御するためには、API認証とデバイス情報の取得を関数化して整理することが効果的です。以下は、環境変数を使用してトークンとシークレットキーを取得し、SwitchBot API v1.1に認証してデバイス情報を取得する方法です。

### 環境設定
セキュリティを確保するために、トークンとシークレットキーは環境変数から取得します。

### ヘッダーの生成
API認証に必要なヘッダーを生成する関数を作成します。

```python

import os
import uuid
import time
import hashlib
import hmac
import base64
import requests

def create_header():
    token = os.environ['SWITCHBOT_TOKEN']
    secret = os.environ['SWITCHBOT_SECRET']
    nonce = uuid.uuid4()
    timestamp = int(round(time.time() * 1000))
    string_to_sign = f'{token}{timestamp}{nonce}'

    signature = base64.b64encode(hmac.new(secret.encode(), msg=string_to_sign.encode(), digestmod=hashlib.sha256).digest()).decode()
    return {
        'Authorization': token,
        'Content-Type': 'application/json',
        'charset': 'utf8',
        't': str(timestamp),
        'sign': signature,
        'nonce': str(nonce)
    }

```

### デバイスリストの取得
```python
def get_devices(header):
    url = 'https://api.switch-bot.com/v1.1/devices'
    response = requests.get(url, headers=header)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.json()}"
```

### 実行
```python
header = create_header()
devices = get_devices(header)
print(devices)
```

### 結果(辞書の出力を整えたもの)
```
{
  "statusCode": 100,
  "body": {
    "deviceList": [
      {
        "deviceId": "XXXXXXXXXXXX",
        "deviceName": "スマート電球 82",
        "deviceType": "Color Bulb",
        "enableCloudService": true,
        "hubDeviceId": ""
      },
      {
        "deviceId": "XXXXXXXXXXXX",
        "deviceName": "ハブ２ BD",
        "deviceType": "Hub 2",
        "enableCloudService": true,
        "hubDeviceId": ""
      }
    ],
    "infraredRemoteList": [
      {
        "deviceId": "XX-XXXXXXXXXXXX-XXXXXXXX",
        "deviceName": "扇風機",
        "remoteType": "DIY Fan",
        "hubDeviceId": "XXXXXXXXXXXXXX"
      }
    ]
  },
  "message": "success"
}
```

bodyのdeviceListとinfraredRemoteListにそれぞれデバイス情報が格納されています。そして、このアカウントにはスマート電球とハブ２、そして扇風機が登録されていることがわかります。
ハブというのはSwitchBotの製品の一つで、赤外線リモコンの信号を学習して、スマートフォンから操作できるようにするものです。ここではそのハブに扇風機の信号を学習させているので、扇風機の情報がinfraredRemoteListに格納されています。
そしてdeviceIdがAPIでデバイスを操作する際に必要になる値です。
今回はこの情報から、デバイスの分類、インスタンスの生成、インスタンスからのデバイス操作を行います。

### 電気をつける
```python
def turn_on(device_id, header):
    url = f'https://api.switch-bot.com/v1.1/devices/{device_id}/commands'
    payload = {
        'command': 'turnOn',# コマンド
        'parameter': 'default'
    }
    response = requests.post(url, headers=header, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.json()}"
```
SwitchBot APIでは家電ごとに登録されているコマンドとデバイスIDの情報で、そのデバイスIDに対してそのコマンドを実行することができます。今回はスマート電球の電気をつけるコマンドを実行します。

### 実行
```python
header = create_header()
devices = get_devices(header)
deviceList = devices['body']['deviceList']
for device in deviceList:
    if device['deviceType'] == 'Color Bulb':
        response=turn_on(device['deviceId'], header)
        print(response)

```

### 結果
```
{'statusCode': 171, 'body': {}, 'message': 'No hub record,is offline'}
```
はい、すみません。オフラインです。うまくいったらうまくいった感じのレスポンスが返ってきます。
ちなみにturnOnやturnOffは電球だけでなく、SwitchBotで操作できる家電のほとんどに対して使えるコマンドです。
コマンド一覧は[こちら](https://github.com/OpenWonderLabs/SwitchBotAPI#command-set-for-virtual-infrared-remote-devices)を参照してください。

## より使いやすくする
デバイスごとにこういうの書いていってもいいんですが、まあ面倒です。そこで、デバイスの種類ごとにクラスを作成して、それぞれのクラスにコマンドを追加していきます。まずは制御するデバイスの共通部分である基底クラスを作成します。

```python
class SwitchBotAppliance:
    """
    SwitchBotのデバイスを操作するための基底クラスです。

    device_id: str
        デバイスID
    device_type: str
        デバイスの種類
    device_name: str
        デバイスの名前
    params: dict
        デバイスのパラメータ、コマンドの引数の説明を格納します。{パラメータ名: 説明}
    """
    def __init__(self, device_id: str, device_type: str, device_name: str):
        self.device_id: str = device_id
        self.device_type: str = device_type
        self.device_name: str = device_name
        self.params = {}

    def turn_on(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の電源を入れます"""
        post_command(self.device_id, "turnOn")

    def turn_off(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の電源を切ります"""
        post_command(self.device_id, "turnOff")

    def make_func_list(self):
        names = dir(self)
        funcs = []
        for name in names:
            if name.startswith("__"):
                continue
            if name.startswith("make_func"):
                continue
            attr = getattr(self, name)
            if callable(attr):
                if not attr.__doc__ == "invalid":
                    funcs.append(method_to_func(attr))
        return funcs
```
make_func_list(とmethod_to_func)については少し後で述べます。

これを継承して、スマート電球のクラスを作成します。
```python
class SwtichBotColorBulb(SwitchBotAppliance):
    def __init__(self, device_id: str, device_type: str, device_name: str):
        super().__init__(device_id, device_type, device_name)
        self.params.update({
            "blue": "青(0-255)",
            "green": "緑(0-255)",
            "red": "赤(0-255)",
            "brightness": "明るさ(0-100)",
        })

    def set_color(self, red: int, green: int, blue: int):
        """{DEVICE_NAME}({DEVICE_TYPE})の色を変更します"""
        post_command(self.device_id, "setColor", f"{red}:{green}:{blue}")

    def set_brightness(self, brightness: int):
        """{DEVICE_NAME}({DEVICE_TYPE})の明るさを変更します"""
        post_command(self.device_id, "setBrightness", str(brightness))
```
こんな感じですね。

```python
bulb = SwtichBotColorBulb(device_id, device_type, device_name)
bulb.turn_on()
bulb.set_color(255, 0, 0)
```
という風に使うことができます。

ここまで来ておかしい要素が出まくってるので一気に説明します。

実は前回の記事の[**自作APIラッパーでChatGPTをしばく**](https://qiita.com/Chroma7p/items/5ba7b7c11aaf6aaa3113)で登場したFunctionオブジェクトを生成するのがmake_func_listです。
make_func_listで各クラスのメソッドをFunctionオブジェクトに変換することができるため、一度クラスを定義してしまえばあとは簡単にChatGPTに接続できるようになります。

```python
import inspect
from chatgpt import GPTFunction, GPTFunctionParam, GPTFunctionProperties #これは自作ライブラリ
def method_to_func(method):
    # メソッド名を取得
    method_name = method.__name__+"_" + \
        method.__self__.device_type.replace(" ", "_")
    # ドキュメントを取得、デバイス名とタイプを置換
    doc = method.__doc__
    doc = doc.replace("{DEVICE_NAME}", method.__self__.device_name)
    doc = doc.replace("{DEVICE_TYPE}", method.__self__.device_type)
    # 引数の情報を取得
    signature = inspect.signature(method)
    require = []
    all_params = []
    param_dic = method.__self__.params
    for name, param in signature.parameters.items():
        param_desc = param_dic.get(name)
        if param_desc is None:
            param_desc = ""
        type_ = param.annotation
        if type_ == str:
            type_ = "string"
        elif type_ == int:
            type_ = "integer"
        elif type_ == bool:
            type_ = "bool"
        else:
            raise TypeError(f"引数の型が不正です: {name} {type_}")
        param_obj = GPTFunctionProperties(name, type_, param_desc)

        if param.default == inspect.Parameter.empty:
            require.append(param_obj)

        all_params.append(param_obj)

    return GPTFunction(method_name, doc, GPTFunctionParam(require, all_params), method)
```

このようにして各クラスのメソッドをFunctionオブジェクトに変換しています。ChatGPTに渡すためのFunctionオブジェクトには、関数名、説明、引数の情報、実際に実行する関数の4つの情報が必要です。これらを取得するために、inspectモジュールを使用して、メソッド名、ドキュメント、引数名の情報を取得しています。また、引数の説明の情報は、メソッドを持つクラスのparamsに書くようにしているので、それを使って引数の説明も付与しています。
これらをFunctionオブジェクトに変換することで、ChatGPTに渡すことができるようになります。
各デバイスのクラスのドキュメントを利用して関数の説明を作っているのはそこそこ邪道だとは思いますが、これでChatGPTにデバイスの操作を教えることができるようになります。
一応大問題ポイントとしてまだおなじクラスから生成した関数の名前がかぶる問題があります。基底クラス弄れば何とかなる気はしていますがまだ実装してません。すみません。

記述量がかさむので、具体的なものはひとりアドベントカレンダーのリポジトリにでも置いておきます。
デバイスリストからFunctionオブジェクトの自動生成はこんな感じになります。
```
{'name': 'turn_off_Color_Bulb', 'description': 'スマート電球 82(Color Bulb)の電源を切ります', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'turn_on_Color_Bulb', 'description': 'スマート電球 82(Color Bulb)の電源を入れます', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'get_room_condition_Hub_2', 'description': 'ハブ２ BD(Hub 2)がある部屋の状態を取得します', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'high_speed_DIY_Fan', 'description': '扇風機(DIY Fan)の風量を強にします', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'low_speed_DIY_Fan', 'description': '扇風機(DIY Fan)の風量を弱にします', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'middle_speed_DIY_Fan', 'description': '扇風機(DIY Fan)の風量を中にします', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'swing_DIY_Fan', 'description': '扇風機(DIY Fan)の首振りを切り替えます', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'timer_DIY_Fan', 'description': '扇風機(DIY Fan)のタイマーを切り替えます', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'turn_off_DIY_Fan', 'description': '扇風機(DIY Fan)の電源を切ります', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
{'name': 'turn_on_DIY_Fan', 'description': '扇風機(DIY Fan)の電源を入れます', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
```

### 実際に対話形式で使ってみる

```
>>電気つけて
Response: {"statusCode":171,"body":{},"message":"No hub record,is offline"}
電気をつけましたにゃん。
```
というように、関数を呼び出すことができます。(電球がオフラインなので実際はつきませんが……)

## まとめ
アドカレ大遅刻どころか昼間(現在11:20)ですね、すみません。  
今回はSwitchBot API v1.1を使用して、Pythonでデバイスを制御する方法を紹介しました。
また、デバイスごとにクラスを作成して、より使いやすくしました。
このコードは元々ハッカソン向けに大急ぎで作った物で、改善の余地どころか直さないといけないところがまだまだあるので、頑張って直して実用できるラインまで持っていきます。
(実はハッカソン向けに作っただけなので、SwitchBotのデバイス常用してなくて、テストがあんまりできてません、今度常用している人に投げつけてテストさせます)

明日は具体的にFunction callingでどんなことができるのかを書いていきます。たぶん。