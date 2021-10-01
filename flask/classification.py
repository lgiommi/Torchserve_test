import torch
import torch.nn as nn
import torch.nn.functional as fun

class ClassifierNN(nn.Module):
    """
    Provide a neural network model for classification.

    Provide a neural network (NN) model for classification.  The NN is a
    simple, fully connected feed-forward network.  The layout of the NN is
    specified at construction time by providing a tuple.  The length of the
    tuple corresponds to the number of network layers (including input and
    output layers).  Each tuple entry specifies the number of nodes in the
    corresponding layer.  The width of the input and output layer must
    correspond to the number of input variables and classes, respectively.

    The non-linear activation function for the hidden layers is relu.  The
    output activation is linear during training and sigmoid in inference mode.
    We use nn.BCELoss() as the loss function during training, as usual for 
    binary classifiers.

    The recommended optimizer is Adam.

    In case you move the classifier to an accelerator (such as a GPU) make sure
    you construct the optimizer after.  Of course, different optimizers and
    loss functions can be used; make sure the implications are understood, in
    particular for the output layer activation (see above).
    """
    def __init__(self,
                 layout=(29,16,8,1),
                 activation=fun.relu):
        super().__init__()
        self.last_save = None
        self.layout = layout
        self.inference_mode = True  # training clients: change this attribute to False
        self.activation = activation
        self.layers = nn.ModuleList()
        for num_nodes, num_nodes_next in zip(self.layout[:-1], self.layout[1:]):
            self.layers.append(nn.Linear(num_nodes, num_nodes_next))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = torch.sigmoid(self.layers[-1](x))
        return x

    def train(self, mode=True):
        super(ClassifierNN, self).train()
        self.inference_mode = False

    def eval(self):
        super(ClassifierNN, self).eval()
        self.inference_mode = True

    def save_weights(self, tag=None, time_stamp=True, directory=None):
        weight_file_path = 'classifier_weights_'
        if tag is not None:
            weight_file_path += '{}_'.format(tag)
        for width in self.layout[:-1]:
            weight_file_path += '{}x'.format(width)
        weight_file_path += '{}'.format(self.layout[-1])
        if time_stamp:
            weight_file_path += '_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
        weight_file_path += '.pt'
        if directory is not None:
            weight_file_path = os.path.join(directory, weight_file_path)

        torch.save(self.state_dict(), weight_file_path)

        self.last_save = weight_file_path
        
        return weight_file_path

    def copy(self):
        return self.state_dict()