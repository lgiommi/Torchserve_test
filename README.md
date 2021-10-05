# Torchserve_test
In this repo you can find the code used for Torchserve testing.
I chose the ttH analysis use case (used also to test MLaaS4HEP) to train and obtained a PyTorch model.
You can find [here](https://github.com/lgiommi/Torchserve_test/blob/main/keys_values_example.txt) the name of the branches and an example of signal and background events.
The definition of the PyTorch model can be found in [classification.py](https://github.com/lgiommi/Torchserve_test/blob/main/classification.py).

The training part of the PyTorch model inside MLaaS4HEP is obtained adding the following code inside [models.py](https://github.com/lgiommi/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/models.py).
```
train_tensor = torch.utils.data.TensorDataset(torch.tensor(X_train).float(),torch.tensor(Y_train, dtype=torch.float))
test_tensor  = torch.utils.data.TensorDataset(torch.tensor(X_test).float(),torch.tensor(Y_test, dtype=torch.float))
eval_tensor  = torch.utils.data.TensorDataset(torch.tensor(X_val).float(),torch.tensor(Y_val, dtype=torch.float))

train_loader = utils_data.DataLoader(train_tensor, batch_size=512, shuffle=True)
test_loader = utils_data.DataLoader(test_tensor, batch_size=512, shuffle=False)

classifier = classification.ClassifierNN(layout=(idim, 16, 8, 1))
h_t, h_e, weights_path = classification.train_classifier(classifier, 
          train_loader, 
          test_loader,
          max_epochs=50,
          learning_rate=0.002,
          momentum=0.9,
          weight_decay=1e-5,
          save_dir='/Users/luca.giommi/Computer_Windows/Universita/Dottorato/TFaaS/MLaaS4HEP/src/python/MLaaS4HEP/weights',
          weight_file_tag=None)
torch.save(classifier, 'model.pt')
```

Then I used [Torch Model archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md) to produce a .mar file.
I launched it in the following way in terminal_1:
```
torch-model-archiver --model-name pytorch_physics_3 --version 1.0 --model-file classification.py --serialized-file pytorch_model_idim_16_8_1.pt --handler handler.py
```
I launch Torchserve in terminal_2:
```
torchserve --start --ncs --model-store model_store
```
I upload the model in terminal_1:
```
curl -X POST "http://localhost:8081/models?initial_workers=1&batch_size=4&url=pytorch_physics_3.mar"
```
obtaining the following output:
```
{
  "status": "Model \"pytorch_physics_3\" Version: 1.0 registered with 1 initial workers"
}
```
Then I ask for prediction for the event stored in [predict_bkg.json](https://github.com/lgiommi/Torchserve_test/blob/main/predict_bkg.json) launching:
```
curl http://localhost:8080/predictions/pytorch_physics_3
```
obtaining:
```
6.588430551346391e-05
```
and for the signal event stored in [predict_signal.json](https://github.com/lgiommi/Torchserve_test/blob/main/predict_signal.json) it gives
```
0.4007512331008911
```
We also obtain that the TorchServe and Flask approach give the same results in terms of predictions.
