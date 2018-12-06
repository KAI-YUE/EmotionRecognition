% CNN Model for CVPR2017
% Change the DirRoot and Options if necessary.

load('train.mat');
load('val.mat');

BValTbl = splitBodies(val);
BTrainTbl = splitBodies(train);

augmentOptions = struct('flip',1,'rotate',0);

bidsTrain = bindImageDatastore([224 224],BTrainTbl,augmentOptions);
bidsVal = bindImageDatastore([224 224],BValTbl,augmentOptions);
Options = loadOptions(bidsVal);
FullNet = InitNet;

trainedNet = trainNetwork(bidsTrain,FullNet,Options);
