import sanic
import httpx
import os
from sanic import Sanic,response
from sanic.response import json
from sanic.websocket import WebSocketProtocol
from sanic.exceptions import NotFound
from sanic.response import html
from jinja2 import Environment, PackageLoader
from sanic_openapi import doc


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
    html_content = template.render(title='AI心理咨询师V1.0')
    return html(html_content)

server_port = int(os.getenv('SERVER_PORT', 12348))
@app.post("/bot/message")
@app.websocket('/chat')
@doc.consumes(doc.JsonBody({"message": str}), location="body")
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

        user_intent = classifier(user_msg)
        print(user_intent)

        if user_intent in ["greet", "goodbye", "deny", "isbot"]:
            reply = chitchat_bot(user_intent)
            print(reply,"aaa")
        elif user_intent == "accept":
            reply = load_user_dialogue_context("qzy")
            reply = reply.get("choice_answer")
            print("01-accept:", reply)
        # diagnosis
        else:
            reply = medical_bot(user_msg)
            print(reply, "bbb")
            if reply["slot_values"]:
                print("ccc")
                dump_user_dialogue_context("qzy",reply)
            print("ddd")
            reply = reply.get("replay_answer")
        print('Sending: ' + reply)
        await ws.send(reply)


if __name__ == "__main__":
    app.error_handler.add(
        NotFound,
        lambda r, e: sanic.response.empty(status=404)
    )
    app.run(host="127.0.0.1", port=8000, protocol=WebSocketProtocol, debug=False)

