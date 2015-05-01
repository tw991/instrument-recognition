require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'

noutputs = 11
batchSize = 100

-- input dimensions
ninputs = nfeats*length

-- hidden units, filter sizes (for ConvNet only):
nstates = {512,512,512,1024}
filtsize = {4, 3, 3, 2}
poolsize = {2, 2, 2, 2}
stridesize = {2, 2, 2, 2}
viewsize = 4

print '==> construct model'

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[1],1, stridesize[1], 1))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[2],1, stridesize[2], 1))

-- stage 2.5
model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize[3], 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[3],1, stridesize[3], 1))
model:add(nn.SpatialConvolutionMM(nstates[3], nstates[3], filtsize[4], 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize[4],1, stridesize[4], 1))
-- stage 3 : 
model:add(nn.Reshape(viewsize*nstates[3]))
model:add(nn.Linear(nstates[3]*viewsize, nstates[4]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[4], nstates[4]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))

-- stage 4:
model:add(nn.Linear(nstates[4], noutputs))
model:add(nn.Sigmoid())
-- loss:
criterion = nn.BCECriterion()

----TEST
--model:add(nn.SpatialConvolutionMM(nfeats, 512, 11, 1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolutionMM(512,512,10,1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolutionMM(512, 512,11,1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolutionMM(512,512,11,1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolutionMM(512,512,10,1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolutionMM(512,512,10,1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolutionMM(512,512,11,1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(2,1,2,1))
--model:add(nn.SpatialConvolutionMM(512,512,11,1))
--model:add(nn.ReLU())
--model:add(nn.SpatialMaxPooling(5,1,5,1))
--model:add(nn.Reshape(65*512))
--model:add(nn.Dropout(0.5))
--model:add(nn.Linear(512*65, 512))
--model:add(nn.ReLU())
--model:add(nn.Linear(512, noutputs))
--model:add(nn.Sigmoid())
--criterion = nn.BCECriterion()
