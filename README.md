# Torchserve_test
In this repo you can find the code used for Torchserve testing
I fix the ttH analysis use case (used also to test MLaaS4HEP) to train and obtained a PyTorch model.
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
torch-model-archiver --model-name pytorch_physics_3 --version 1.0 --model-file classification.py --serialized-file pytorch_model_idim_16_8_1.pt --extra-files index_to_name.json --handler base_handler.py
```
I launch Torchserve in terminal_2:
```
torchserve --start --ncs --model-store model_store
```
I try to upload the model in terminal_1:
```
curl -X POST "http://localhost:8081/models?initial_workers=1&batch_size=4&url=pytorch_physics_5.mar"
```
But I got the following error in terminal_2:
```
2021-07-20 15:13:36,380 [DEBUG] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelVersionedRefs - Adding new version 1.0 for model pytorch_physics_4
2021-07-20 15:13:36,381 [DEBUG] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelVersionedRefs - Setting default version to 1.0 for model pytorch_physics_4
2021-07-20 15:13:36,381 [INFO ] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelManager - Model pytorch_physics_4 loaded.
2021-07-20 15:13:36,382 [DEBUG] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelManager - updateModel: pytorch_physics_4, count: 1
2021-07-20 15:13:39,709 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - Listening on port: /var/folders/c2/9kbjprj514zbk6pyd3dg7t0c0000gr/T//.ts.sock.9000
2021-07-20 15:13:39,711 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - [PID]39682
2021-07-20 15:13:39,711 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - Torch worker started.
2021-07-20 15:13:39,712 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - Python runtime: 3.8.8
2021-07-20 15:13:39,715 [DEBUG] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerThread - W-9000-pytorch_physics_4_1.0 State change null -> WORKER_STARTED
2021-07-20 15:13:39,739 [INFO ] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerThread - Connecting to: /var/folders/c2/9kbjprj514zbk6pyd3dg7t0c0000gr/T//.ts.sock.9000
2021-07-20 15:13:39,795 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - Connection accepted: /var/folders/c2/9kbjprj514zbk6pyd3dg7t0c0000gr/T//.ts.sock.9000.
2021-07-20 15:13:39,874 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - Backend worker process died.
2021-07-20 15:13:39,875 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - Traceback (most recent call last):
2021-07-20 15:13:39,876 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 182, in <module>
2021-07-20 15:13:39,876 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -     worker.run_server()
2021-07-20 15:13:39,877 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 154, in run_server
2021-07-20 15:13:39,877 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -     self.handle_connection(cl_socket)
2021-07-20 15:13:39,877 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 116, in handle_connection
2021-07-20 15:13:39,878 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -     service, result, code = self.load_model(msg)
2021-07-20 15:13:39,880 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 89, in load_model
2021-07-20 15:13:39,881 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -     service = model_loader.load(model_name, model_dir, handler, gpu, batch_size, envelope)
2021-07-20 15:13:39,882 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_loader.py", line 83, in load
2021-07-20 15:13:39,882 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -     module = self._load_default_handler(handler)
2021-07-20 15:13:39,882 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_loader.py", line 126, in _load_default_handler
2021-07-20 15:13:39,883 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -     module = importlib.import_module(module_name, 'ts.torch_handler')
2021-07-20 15:13:39,884 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/importlib/__init__.py", line 127, in import_module
2021-07-20 15:13:39,884 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -     return _bootstrap._gcd_import(name[level:], package, level)
2021-07-20 15:13:39,884 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
2021-07-20 15:13:39,884 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "<frozen importlib._bootstrap>", line 991, in _find_and_load
2021-07-20 15:13:39,880 [INFO ] KQueueEventLoopGroup-5-1 org.pytorch.serve.wlm.WorkerThread - 9000 Worker disconnected. WORKER_STARTED
2021-07-20 15:13:39,884 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG -   File "<frozen importlib._bootstrap>", line 970, in _find_and_load_unlocked
2021-07-20 15:13:39,886 [INFO ] W-9000-pytorch_physics_4_1.0-stdout MODEL_LOG - ModuleNotFoundError: No module named 'ts.torch_handler.base_handler.py'; 'ts.torch_handler.base_handler' is not a package
2021-07-20 15:13:39,886 [DEBUG] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerThread - System state is : WORKER_STARTED
2021-07-20 15:13:39,889 [DEBUG] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerThread - Backend worker monitoring thread interrupted or backend worker process died.
java.lang.InterruptedException
	at java.base/java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject.awaitNanos(AbstractQueuedSynchronizer.java:1668)
	at java.base/java.util.concurrent.ArrayBlockingQueue.poll(ArrayBlockingQueue.java:435)
	at org.pytorch.serve.wlm.WorkerThread.run(WorkerThread.java:188)
	at java.base/java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:515)
	at java.base/java.util.concurrent.FutureTask.run(FutureTask.java:264)
	at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1130)
	at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:630)
	at java.base/java.lang.Thread.run(Thread.java:832)
2021-07-20 15:13:39,901 [WARN ] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.BatchAggregator - Load model failed: pytorch_physics_4, error: Worker died.
2021-07-20 15:13:39,901 [DEBUG] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.ModelVersionedRefs - Removed model: pytorch_physics_4 version: 1.0
2021-07-20 15:13:39,901 [DEBUG] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerThread - W-9000-pytorch_physics_4_1.0 State change WORKER_STARTED -> WORKER_SCALED_DOWN
2021-07-20 15:13:39,901 [WARN ] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_4_1.0-stderr
2021-07-20 15:13:39,902 [WARN ] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_4_1.0-stdout
2021-07-20 15:13:39,929 [INFO ] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.ModelManager - Model pytorch_physics_4 unregistered.
2021-07-20 15:13:39,930 [DEBUG] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerThread - W-9000-pytorch_physics_4_1.0 State change WORKER_SCALED_DOWN -> WORKER_STOPPED
2021-07-20 15:13:39,930 [INFO ] W-9000-pytorch_physics_4_1.0-stderr org.pytorch.serve.wlm.WorkerLifeCycle - Stopped Scanner - W-9000-pytorch_physics_4_1.0-stderr
2021-07-20 15:13:39,930 [INFO ] W-9000-pytorch_physics_4_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - Stopped Scanner - W-9000-pytorch_physics_4_1.0-stdout
2021-07-20 15:13:39,930 [WARN ] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_4_1.0-stderr
2021-07-20 15:13:39,931 [WARN ] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_4_1.0-stdout
2021-07-20 15:13:39,931 [DEBUG] W-9000-pytorch_physics_4_1.0 org.pytorch.serve.wlm.WorkerThread - Worker terminated due to scale-down call.
2021-07-20 15:13:39,945 [INFO ] KQueueEventLoopGroup-3-1 ACCESS_LOG - /127.0.0.1:50382 "POST /models?initial_workers=1&batch_size=4&url=pytorch_physics_4.mar HTTP/1.1" 500 3647
2021-07-20 15:13:39,946 [INFO ] KQueueEventLoopGroup-3-1 TS_METRICS - Requests5XX.Count:1|#Level:Host|#hostname:str957-135.local,timestamp:null
```
And in terminal_1:
```
{
  "code": 500,
  "type": "InternalServerException",
  "message": "Failed to start workers for model pytorch_physics_6 version: 1.0"
}
```
I tried to use a simpler handler file [hanlder.py](https://github.com/lgiommi/Torchserve_test/blob/main/handler.py) launching torch-model-archiver but now the error is:
```
2021-07-20 15:36:22,356 [DEBUG] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelVersionedRefs - Adding new version 1.0 for model pytorch_physics_7
2021-07-20 15:36:22,359 [DEBUG] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelVersionedRefs - Setting default version to 1.0 for model pytorch_physics_7
2021-07-20 15:36:22,360 [INFO ] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelManager - Model pytorch_physics_7 loaded.
2021-07-20 15:36:22,361 [DEBUG] KQueueEventLoopGroup-3-1 org.pytorch.serve.wlm.ModelManager - updateModel: pytorch_physics_7, count: 1
2021-07-20 15:36:26,377 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - Listening on port: /var/folders/c2/9kbjprj514zbk6pyd3dg7t0c0000gr/T//.ts.sock.9000
2021-07-20 15:36:26,378 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - [PID]40164
2021-07-20 15:36:26,378 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - Torch worker started.
2021-07-20 15:36:26,379 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - Python runtime: 3.8.8
2021-07-20 15:36:26,379 [DEBUG] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerThread - W-9000-pytorch_physics_7_1.0 State change null -> WORKER_STARTED
2021-07-20 15:36:26,391 [INFO ] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerThread - Connecting to: /var/folders/c2/9kbjprj514zbk6pyd3dg7t0c0000gr/T//.ts.sock.9000
2021-07-20 15:36:26,405 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - Connection accepted: /var/folders/c2/9kbjprj514zbk6pyd3dg7t0c0000gr/T//.ts.sock.9000.
2021-07-20 15:36:26,554 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - Backend worker process died.
2021-07-20 15:36:26,555 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - Traceback (most recent call last):
2021-07-20 15:36:26,556 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 182, in <module>
2021-07-20 15:36:26,556 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -     worker.run_server()
2021-07-20 15:36:26,558 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 154, in run_server
2021-07-20 15:36:26,558 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -     self.handle_connection(cl_socket)
2021-07-20 15:36:26,558 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 116, in handle_connection
2021-07-20 15:36:26,558 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -     service, result, code = self.load_model(msg)
2021-07-20 15:36:26,559 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_service_worker.py", line 89, in load_model
2021-07-20 15:36:26,560 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -     service = model_loader.load(model_name, model_dir, handler, gpu, batch_size, envelope)
2021-07-20 15:36:26,562 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/ts/model_loader.py", line 110, in load
2021-07-20 15:36:26,562 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -     initialize_fn(service.context)
2021-07-20 15:36:26,562 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -   File "/private/var/folders/c2/9kbjprj514zbk6pyd3dg7t0c0000gr/T/models/18516dabea0247d097aa74d3603f5ca3/handler.py", line 35, in initialize
2021-07-20 15:36:26,563 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -     self.model = torch.jit.load(model_pt_path)
2021-07-20 15:36:26,563 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -   File "/Users/luca.giommi/anaconda3/envs/tensorflow/lib/python3.8/site-packages/torch/jit/_serialization.py", line 161, in load
2021-07-20 15:36:26,563 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG -     cpp_module = torch._C.import_ir_module(cu, str(f), map_location, _extra_files)
2021-07-20 15:36:26,563 [INFO ] W-9000-pytorch_physics_7_1.0-stdout MODEL_LOG - RuntimeError: istream reader failed: reading file.
2021-07-20 15:36:26,561 [INFO ] KQueueEventLoopGroup-5-1 org.pytorch.serve.wlm.WorkerThread - 9000 Worker disconnected. WORKER_STARTED
2021-07-20 15:36:26,565 [DEBUG] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerThread - System state is : WORKER_STARTED
2021-07-20 15:36:26,567 [DEBUG] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerThread - Backend worker monitoring thread interrupted or backend worker process died.
java.lang.InterruptedException
	at java.base/java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject.awaitNanos(AbstractQueuedSynchronizer.java:1668)
	at java.base/java.util.concurrent.ArrayBlockingQueue.poll(ArrayBlockingQueue.java:435)
	at org.pytorch.serve.wlm.WorkerThread.run(WorkerThread.java:188)
	at java.base/java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:515)
	at java.base/java.util.concurrent.FutureTask.run(FutureTask.java:264)
	at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1130)
	at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:630)
	at java.base/java.lang.Thread.run(Thread.java:832)
2021-07-20 15:36:26,576 [WARN ] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.BatchAggregator - Load model failed: pytorch_physics_7, error: Worker died.
2021-07-20 15:36:26,577 [DEBUG] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.ModelVersionedRefs - Removed model: pytorch_physics_7 version: 1.0
2021-07-20 15:36:26,578 [DEBUG] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerThread - W-9000-pytorch_physics_7_1.0 State change WORKER_STARTED -> WORKER_SCALED_DOWN
2021-07-20 15:36:26,578 [WARN ] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_7_1.0-stderr
2021-07-20 15:36:26,578 [WARN ] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_7_1.0-stdout
2021-07-20 15:36:26,612 [INFO ] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.ModelManager - Model pytorch_physics_7 unregistered.
2021-07-20 15:36:26,613 [DEBUG] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerThread - W-9000-pytorch_physics_7_1.0 State change WORKER_SCALED_DOWN -> WORKER_STOPPED
2021-07-20 15:36:26,616 [WARN ] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_7_1.0-stderr
2021-07-20 15:36:26,625 [WARN ] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerLifeCycle - terminateIOStreams() threadName=W-9000-pytorch_physics_7_1.0-stdout
2021-07-20 15:36:26,625 [DEBUG] W-9000-pytorch_physics_7_1.0 org.pytorch.serve.wlm.WorkerThread - Worker terminated due to scale-down call.
2021-07-20 15:36:26,630 [INFO ] W-9000-pytorch_physics_7_1.0-stderr org.pytorch.serve.wlm.WorkerLifeCycle - Stopped Scanner - W-9000-pytorch_physics_7_1.0-stderr
2021-07-20 15:36:26,626 [INFO ] W-9000-pytorch_physics_7_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - Stopped Scanner - W-9000-pytorch_physics_7_1.0-stdout
2021-07-20 15:36:26,637 [INFO ] KQueueEventLoopGroup-3-1 ACCESS_LOG - /127.0.0.1:50751 "POST /models?initial_workers=1&batch_size=4&url=pytorch_physics_7.mar HTTP/1.1" 500 4332




