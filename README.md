# Torchserve_test
In this repo you can find the code used for Torchserve testing
I fix the ttH analysis use case (used also to test MLaaS4HEP) to train and obtained a PyTorch model.
The definition of the PyTorch model can be found in [classification.py](https://github.com/lgiommi/Torchserve_test/blob/main/classification.py).

The training part inside MLaaS4HEP is reached adding the following code inside [models.py](https://github.com/lgiommi/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/models.py).
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
I launched it in the following way
```
torch-model-archiver --model-name pytorch_physics_3 --version 1.0 --model-file classification.py --serialized-file pytorch_model_idim_16_8_1.pt --handler ../torchserve/serve/ts/torch_handler/base_handler.py
