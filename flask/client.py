import requests
load = requests.post("http://localhost:5000/upload", files={"model": open('../pytorch_model.pt','rb')})
resp = requests.post("http://localhost:5000/predict", files={"input": open('../predict.json','r')}, data={'model':'pytorch_model.pt'})
print(resp.text)