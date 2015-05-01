require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> configuring optimizer'


save = '.'
maxEpoch = 400

-- Log results to files
trainLogger = optim.Logger(paths.concat(save, 'train.log'))
testLogger = optim.Logger(paths.concat(save, 'test.log'))

model:float()
criterion:float()

if opt.type == 'cuda' then 
  model:cuda()
  criterion:cuda()
end

print '==> defining training procedure'
parameters, gradParameters = model:getParameters()

function train()
   --shuffle = torch.randperm(trsize)
   -- epoch tracker
   epoch = epoch or 1
   -- local vars
   local time = sys.clock()
   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()
   
   local tloss = 0
   local correct = 0
   local exact_correct = 0
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainData.size,batchSize do
      -- disp progress
      xlua.progress(t, trainData.size)
      -- create mini batch
      if batchSize == 1 then
        inputs = trainData.data[{{shuffle[t]},{}}]
        targets = trainData.labels[{{shuffle[t]},{}}]
      else
        --inputs = trainData.data[{{shuffle[{{t, math.min(t+batchSize-1,trainData.size)}}]}, {}}]
        inputs = trainData.data[{{t, math.min(t+batchSize-1, trainData.size)},{}}]
        targets = trainData.labels[{{t, math.min(t+batchSize-1, trainData.size)},{}}]
        --targets = trainData.labels[{{shuffle[{{t, math.min(t+batchSize-1,trainData.size)}}]}, {}}]
      end
      gradParameters:zero()
      if opt.type == 'cuda' then
        inputs = inputs:cuda()
        targets = targets:cuda()
      end
      local output = model:forward(inputs)
      local loss = criterion:forward(output, targets)
      local out_predict = output:ge(0.5)
      tloss = tloss + loss
      correct = correct + output:ge(0.5):eq(targets:ge(0.5)):sum()
      exact_correct = exact_correct + (out_predict:eq(targets:ge(0.5)):sum(2)):ge(11):sum()
      model:backward(inputs, criterion:backward(output, targets))
      parameters:add(-optimState.learningRate, gradParameters)
      collectgarbage()
    end

   -- time taken
   time = sys.clock() - time
   time = time / trainData.size
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print("\n==> training accuracy %:")
   print(correct / trainData.size / noutputs * 100)
   print("\n==> exact training accuracy %")
   print(exact_correct / trainData.size * 100) 
   print("\n==>training loss")
   print(tloss / trainData.size)

   -- update logger/plot
   trainLogger:add{['% class accuracy (train set)'] = correct / trainData.size / noutputs * 100, ['training loss'] = tloss / trainData.size}

   -- save/log current net
   local filename = paths.concat(save, 'model_large.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   if opt.save == 'True' and epoch %20 == 0 then
     torch.save(filename, model)
   end
   -- next epoch
   epoch = epoch + 1
end

print('==> defining test procedure')

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')

   local tloss = 0
   local correct = 0
   local exact_correct = 0 
   local n_tp = 0 --true positive
   local n_pred = 0 --number of prediction
   local n_pos = 0 --number of target 
   local testBatchSize = 8
      -- disp progress
   for t = 1,testData.size,testBatchSize do
      -- disp progress
      xlua.progress(t, testData.size)

      local input = testData.data[{{t, math.min(t+testBatchSize-1,testData.size)}, {}}]
      local target = testData.labels[{{t, math.min(t+testBatchSize-1,testData.size)}, {}}]
      if opt.type == 'cuda' then
        input = input:cuda()
        target = target:cuda()
      end
      -- test sample
      local pred = model:forward(input)
      local loss = criterion:forward(pred, target)
      local out_predict = pred:ge(0.5)
      n_tp = n_tp + pred:ge(0.5):cmul(target:ge(0.5)):sum()
      n_pred = n_pred + pred:ge(0.5):sum()
      n_pos = n_pos + target:ge(0.5):sum()
      tloss = tloss + loss
      correct = correct + pred:ge(0.5):eq(target:ge(0.5)):sum()
      exact_correct = exact_correct + (out_predict:eq(target:ge(0.5)):sum(2)):ge(11):sum()
      -- print("\n" .. target .. "\n")

   end

   -- timing
   time = sys.clock() - time
   time = time / testData.size
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print('\n Test Accuracy %:')
   print(correct / testData.size / noutputs * 100)
   print("\n==> exact testing accuracy %")
   print(exact_correct/testData.size * 100)
   print("\n==> Test precision")
   print(n_tp/ n_pred)
   print("\n==> Test recall")
   print(n_tp / n_pos)
   print("\n==> Test F micro")
   print(2 * n_tp / (n_pred + n_pos))
   print('\ntest loss:')
   print(tloss / testData.size)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = correct / testData.size / noutputs * 100, ['test loss'] = tloss / testData.size}   
   -- next iteration:

end


