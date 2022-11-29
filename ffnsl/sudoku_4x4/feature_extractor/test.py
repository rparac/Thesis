import torch
import torch.nn as nn
import json
from network import MNISTNet
from dataset import load_data
import numpy as np

torch.set_printoptions(sci_mode=False)

# Instantiate network and load trained weights
net = MNISTNet()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
network_state_dict = torch.load('saved_model/model.pth', map_location=dev)
# network_state_dict = torch.load('saved_model/model.pth', map_location=dev)
net.load_state_dict(network_state_dict)
net.eval()

test_set = 'rotated' # 'standard'
base_dir = f'../cache/digit_predictions/softmax'
# Load test data
_, test_loader = load_data(data_type=test_set)

# Obtain predictions for the test data
correct = 0
preds = {}
preds_old = {}
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(dev)
        target = target.to(dev)
        output = net(data)
        softmax_fn = nn.Softmax(dim=1)
        softmax_output = softmax_fn(output)
        prob = softmax_output.squeeze().tolist()
        preds[str(batch_idx) + '.jpg'] = prob
        preds_old[str(batch_idx) + '.jpg'] = (int(np.argmax(prob)) + 1, float(np.max(prob)))

        # Print first prediction in the batch for now
        print('Softmax output: ', softmax_output[0].tolist())
        print('Softmax prediction confidence: ', max(softmax_output[0].tolist()))
        print('Prediction: ', np.argmax(softmax_output[0].tolist()))
        print('Target: ', target[0])
        print('-----')
        pred = softmax_output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    print('\nTest set. Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                           100. * correct / len(test_loader.dataset)))


print("Saving to cache.")
cache_file = f'{base_dir}/{test_set}_test_set.json'
with open(cache_file, 'w') as cache_out:
    cache_out.write(json.dumps(preds))
cache_file = f'{base_dir}/{test_set}_test_set_old.json'
with open(cache_file, 'w') as cache_out:
    cache_out.write(json.dumps(preds_old))
