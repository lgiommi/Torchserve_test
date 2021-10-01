# Flask app: how to use it
A first attempt of flask app is stored in this folder.

It contains a [server.py](https://github.com/lgiommi/Torchserve_test/blob/main/flask/server.py) that defines the functionalities of the server, and [client.py](https://github.com/lgiommi/Torchserve_test/blob/main/flask/client.py)
with the requests made by the client. Substantially if allows to upload a model and retrieve prediction for a test event.
You can run the server with:
```
FLASK_ENV=development FLASK_APP=server.py flask run
```
and in the client terminal run the [client.py](https://github.com/lgiommi/Torchserve_test/blob/main/flask/client.py) script with
```
python client.py
```
obtaining the prediction as output.
