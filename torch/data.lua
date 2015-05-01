require 'cutorch'
require 'nn'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')
file = 'train.t7'

print '==> loading dataset'
nfeats = 180
length = 87
loaded = torch.load(file)
trsize = (#loaded['train_y'])[1]
trainData = {
   data = loaded['train_X']:reshape(trsize, nfeats, 1, length),
   labels = loaded['train_y'],
   present = loaded['train_p'],
   size = trsize
}
loaded = nil

file = 'test.t7'
loaded = torch.load(file)
tesize = (#loaded['test_y'])[1]
testData = {
   data = loaded['test_X']:reshape(tesize, nfeats, 1, length),
   labels = loaded['test_y'],
   present = loaded['train_p'],
   size = tesize
}
loaded = nil
collectgarbage()

print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i = 1,nfeats do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i = 1,nfeats do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

for i = 1,nfeats do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..i..'-channel, mean: ' .. trainMean)
   print('training data, '..i..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..i..'-channel, mean: ' .. testMean)
   print('test data, '..i..'-channel, standard deviation: ' .. testStd)
end
