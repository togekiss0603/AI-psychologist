from sanic import Sanic
import os
from sanic import response
from sanic.response import json, redirect, text
from sanic.websocket import WebSocketProtocol

app = Sanic("server")
# ©YDOOK JYLin

@app.route("/", methods=['GET','POST'])
async def home(request):
    return text('home')

# ©YDOOK JYLin
@app.route("/get_pic", methods=['GET','POST'])
async def get_pic(request):
    if request.method == 'POST':
        f = request.files.get("img")
        print(f.name)
        print(f.type)
        print(f.body)
        with open('./wanye.jpg', 'wb') as fileUp:  # 这里必须为读写二进制模式的 wb
            fileUp.write(f.body)
            fileUp.close()
        return text('get_pic')


app.run(host="127.0.0.1", port=8000,protocol = WebSocketProtocol, debug=False)