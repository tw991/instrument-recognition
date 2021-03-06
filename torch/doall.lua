require 'torch'
require 'cutorch'
print '==> processing options'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-type','cuda','type: float | cuda')
cmd:option('-save','False','save: False | True')
cmd:option('-machine','k80','machine: k80 | hpc')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
if opt.machine == 'k80' then 
  cutorch.setDevice(3)
end

optimState = {
    learningRate = 0.1,
    momentum = 0.95,
    learningRateDecay = 2,
    max_epoch = 20
}
print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
dofile 'class.lua'
dofile 'train.lua'

epoch = 1

while epoch < maxEpoch do
   if epoch % optimState.max_epoch == 0 then
      optimState.learningRate = optimState.learningRate / optimState.learningRateDecay
   end
   train()
   collectgarbage()
   test()
   collectgarbage()
end
