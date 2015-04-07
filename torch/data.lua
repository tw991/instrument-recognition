require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')
file = 'train.t7'

print '==> loading dataset'

loaded = torch.load(file)
trsize = (#loaded['train_y'])[1]
trainData = {
   data = loaded['train_X']:reshape(trsize, 2, 1, 44100),
   labels = loaded['train_y'],
   size = trsize
}
loaded = nil

file = 'test.t7'
loaded = torch.load(file)
tesize = (#loaded['test_y'])[1]
testData = {
   data = loaded['test_X']:reshape(tesize, 2, 1, 44100),
   labels = loaded['test_y'],
   size = tesize
}

loaded = nil

collectgarbage()
