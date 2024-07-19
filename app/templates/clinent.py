import requests,json

url = 'http://127.0.0.1:8000'

filesD = {'img': ('1.jpg', open('wanye.jpg', 'rb'), 'image/jpeg')}

# ©YDOOK JYLin
r = requests.post(url=url, files=filesD)

print('r = ', r)
print('r.text = ', r.text)
print('r.content = ', r.content)