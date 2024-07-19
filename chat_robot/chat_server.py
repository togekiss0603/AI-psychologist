import sanic
import httpx
from sanic import Sanic,response
from sanic.response import json
from sanic.websocket import WebSocketProtocol
from sanic.exceptions import NotFound
from sanic.response import html
from jinja2 import Environment, PackageLoader

from modules import chitchat_bot,medical_bot, classifier
from utils.json_utils import dump_user_dialogue_context, load_user_dialogue_context

env = Environment(loader=PackageLoader('app', 'templates'))

app = Sanic(__name__)


@app.route('/')
async def index(request):
    """
    聊天页面
    """
    template = env.get_template('index.html')
    html_content = template.render(title='聊天机器人')
    return html(html_content)

@app.websocket('/chat')
async def chat(request, ws):
    """
    处理聊天信息，并返回消息
    :param request:
    :param ws:
    :return:
    """
    while True:
        user_msg = await ws.recv()
        print('Received: ' + user_msg)
        intelligence_data = {"key": "free", "appid": 0, "msg": user_msg}

        sender = request.json.get("sender")

        user_intent = classifier(user_msg)
        print("user_intent:", user_intent)
        if user_intent in ["greet", "goodbye", "deny", "isbot"]:
            reply = chitchat_bot(user_intent)
        elif user_intent == "accept":
            reply = load_user_dialogue_context(sender)
            reply = reply.get("choice_answer")
            print("01-accept:", reply)
        # diagnosis
        else:
            reply = medical_bot(user_msg, sender)
            if reply["slot_values"]:
                dump_user_dialogue_context(sender, reply)
            reply = reply.get("replay_answer")
        print('Sending: ' + reply)
        await ws.send(reply)


if __name__ == "__main__":
    app.error_handler.add(
        NotFound,
        lambda r, e: sanic.response.empty(status=404)
    )
    app.run(host="127.0.0.1", port=8000, protocol=WebSocketProtocol, debug=True)

