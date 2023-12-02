import json
import time
import hashlib
import hmac
import base64
import requests
from chatgpt import Chat, Message, Role, Response, GPTFunction, GPTFunctionParam, GPTFunctionProperties, Model
import os
from dotenv import load_dotenv
import inspect
load_dotenv()

API_TOKEN = os.getenv("OPENAI_API_KEY")


API_BASE_URL = "https://api.switch-bot.com"
ACCESS_TOKEN = os.getenv("SWITCHBOT_ACCESS_TOKEN")
SECRET = os.getenv("SWITCHBOT_SECRET")


def generate_sign(token: str, secret: str, nonce: str) -> tuple[str, str]:
    """SWITCH BOT APIの認証キーを生成する"""

    t = int(round(time.time() * 1000))
    string_to_sign = "{}{}{}".format(token, t, nonce)
    string_to_sign_b = bytes(string_to_sign, "utf-8")
    secret_b = bytes(secret, "utf-8")
    sign = base64.b64encode(
        hmac.new(secret_b, msg=string_to_sign_b,
                 digestmod=hashlib.sha256).digest()
    )

    return (str(t), str(sign, "utf-8"))


def get_device_list() -> str:
    """SWITCH BOTのデバイスリストを取得する"""

    nonce = "zzz"
    t, sign = generate_sign(ACCESS_TOKEN, SECRET, nonce)
    headers = {
        "Authorization": ACCESS_TOKEN,
        "t": t,
        "sign": sign,
        "nonce": nonce,
    }
    url = f"{API_BASE_URL}/v1.1/devices"
    r = requests.get(url, headers=headers)

    return json.dumps(r.json(), indent=2, ensure_ascii=False)


def post_command(
    device_id: str,
    command: str,
    parameter: str = "default",
    command_type: str = "command",
) -> requests.Response:
    """指定したデバイスにコマンドを送信する"""

    nonce = "zzz"
    t, sign = generate_sign(ACCESS_TOKEN, SECRET, nonce)
    headers = {
        "Content-Type": "application/json; charset: utf8",
        "Authorization": ACCESS_TOKEN,
        "t": t,
        "sign": sign,
        "nonce": nonce,
    }
    if command == "get_room_condition":
        url = f"{API_BASE_URL}/v1.1/devices/{device_id}/status"
        try:
            # print(f"Post command: {data}")
            r = requests.post(url, headers=headers)
            print(f"Response: {r.text}")
        except requests.exceptions.RequestException as e:
            print(e)
            pass

    else:
        url = f"{API_BASE_URL}/v1.1/devices/{device_id}/commands"
        data = json.dumps(
            {"command": command, "parameter": parameter, "commandType": command_type}
        )
        try:
            # print(f"Post command: {data}")
            r = requests.post(url, data=data, headers=headers)
            print(f"Response: {r.text}")
        except requests.exceptions.RequestException as e:
            print(e)
            pass

    return r


def method_to_func(method):
    method_name = method.__name__+"_" + \
        method.__self__.device_type.replace(" ", "_")
    doc = method.__doc__
    doc = doc.replace("{DEVICE_NAME}", method.__self__.device_name)
    doc = doc.replace("{DEVICE_TYPE}", method.__self__.device_type)
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

    execute_func = lambda *args, **kwargs: method(*args, **kwargs)
    return GPTFunction(method_name, doc, GPTFunctionParam(all_params, require), method)


class SwitchBotAppliance:
    def __init__(self, device_id: str, device_type: str, device_name: str):
        self.device_id: str = device_id
        self.device_type: str = device_type
        self.device_name: str = device_name
        self.params = {
            # "button_name": "ボタンの名前",
        }

    def turn_on(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の電源を入れます"""
        post_command(self.device_id, "turnOn")

    def turn_off(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の電源を切ります"""
        post_command(self.device_id, "turnOff")

    # def custom_button(self,button_name:str):
        """
        {DEVICE_NAME}({DEVICE_TYPE})に設定されている任意のボタンを押します
        
        Args:
            button_name (str): ボタンの名前

        """
        # post_command(self.device_id,button_name)

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


class SwitchBotTV(SwitchBotAppliance):
    def __init__(self, device_id: str, device_type: str, device_name: str):
        super().__init__(device_id, device_type, device_name)
        self.params.update({
            "channel": "チャンネル",
            "volume": "音量",
        })

    def set_channel(self, channel: int):
        """
        テレビのチャンネルを変更します

        Args:
            channel (int): チャンネル
        """
        post_command(self.device_id, "setChannel", str(channel))

    def volume_add(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の音量を上げます"""
        post_command(self.device_id, "volumeAdd")

    def volume_sub(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の音量を下げます"""
        post_command(self.device_id, "volumeSub")

    def channel_add(self):
        """{DEVICE_NAME}({DEVICE_TYPE})のチャンネル番号を増やします"""
        post_command(self.device_id, "channelAdd")

    def channel_sub(self):
        """{DEVICE_NAME}({DEVICE_TYPE})のチャンネル番号を減らします"""
        post_command(self.device_id, "channelSub")


class SwitchBotAirConditioner(SwitchBotAppliance):
    def __init__(self, device_id: str, device_type: str, device_name: str):
        super().__init__(device_id, device_type, device_name)
        self.params.update({
            "temperature": "温度",
            "mode": "モード、1:自動,2:冷房,3:除湿,4:送風,5:暖房",
            "fan_speed": "風量、1:自動,2:弱,3:中,4:強",
            "power": "電源、onもしくはoff",
        })

    def set_all(self, temperature: int, mode: int, fan_speed: int, power: str):
        """{DEVICE_NAME}({DEVICE_TYPE})の設定を変更します"""
        txt = f"{temperature},{mode},{fan_speed},{power}"
        post_command(self.device_id, "setAll", txt)


class SwitchBotFan(SwitchBotAppliance):
    def __init__(self, device_id: str, device_type: str, device_name: str):
        super().__init__(device_id, device_type, device_name)

    def swing(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の首振りを切り替えます"""
        post_command(self.device_id, "swing")

    def timer(self):
        """{DEVICE_NAME}({DEVICE_TYPE})のタイマーを切り替えます"""
        post_command(self.device_id, "timer")

    def low_speed(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の風量を弱にします"""
        post_command(self.device_id, "lowSpeed")

    def middle_speed(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の風量を中にします"""
        post_command(self.device_id, "middleSpeed")

    def high_speed(self):
        """{DEVICE_NAME}({DEVICE_TYPE})の風量を強にします"""
        post_command(self.device_id, "highSpeed")


class SwitchBotHub2(SwitchBotAppliance):
    def __init__(self, device_id: str, device_type: str, device_name: str):
        super().__init__(device_id, device_type, device_name)

    def turn_on(self):
        """invalid"""

    def turn_off(self):
        """invalid"""

    def get_room_condition(self):
        """{DEVICE_NAME}({DEVICE_TYPE})がある部屋の状態を取得します"""
        post_command(self.device_id, "get_room_condition")


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


def devicelist_to_instance_list(device_list):
    instances = []
    for device in device_list["body"]["deviceList"]:
        device_id = device["deviceId"]
        device_type = device["deviceType"]
        device_name = device["deviceName"]
        if device_type == "Hub 2":
            instances.append(SwitchBotHub2(
                device_id, device_type, device_name))
        elif device_type == "Color Bulb":
            instances.append(SwtichBotColorBulb(
                device_id, device_type, device_name))
    for device in device_list["body"]["infraredRemoteList"]:
        device_id = device["deviceId"]
        device_type = device["remoteType"]
        device_name = device["deviceName"]
        if device_type == "TV":
            instances.append(SwitchBotTV(device_id, device_type, device_name))
        elif device_type == "Air Conditioner":
            instances.append(SwitchBotAirConditioner(
                device_id, device_type, device_name))
        elif device_type == "DIY Fan":
            instances.append(SwitchBotFan(device_id, device_type, device_name))
    return instances


base_prompt = """\
あなたは家電を操作する事ができるAIアシスタントです。
ユーザーの要望に応じて適切に家電を操作します。
ついでにねこ語で喋りますにゃん。
"""

base_prompt = Message(base_prompt, Role.system)

device_list = json.loads(get_device_list())
print(get_device_list())
appliance_list = devicelist_to_instance_list(device_list)

chat = Chat(API_TOKEN=API_TOKEN, base_prompts=[base_prompt],
            model=Model.gpt35, functions=[])

for appliance in appliance_list:
    chat.functions += appliance.make_func_list()


for x in chat.functions:
    print(x.to_json())
while 1:
    inp = input(">>")
    reply = chat.send_stream(inp)
    for chunk in reply:
        print(chunk, end="")
    print()

