% This function initializes the full network and connets the branches
% of the Model proposed in EMOTIC, CVPR 2017.
% Yue-Kai, USTC.

function FullNet  = InitNet
params = parameters;
Options = struct('wi',params.w,'theta',params.theta,'lamda',params.lamda);
BoLayers = Body_Layers;
ImLayers = Image_Layers;

% Initialize the layerGraph
FullNet = layerGraph(imageInputLayer([224 224 6],'Name','Input',...
                    'Normalization','none'));

% Add the seperate layers to DAG and connect them to the inputLayer
FullNet = addLayers(FullNet,BoLayers);
FullNet = connectLayers(FullNet,'Input','Bbranch');
FullNet = addLayers(FullNet,ImLayers);
FullNet = connectLayers(FullNet,'Input','Ibranch');


% Connet the seperate layers with Output Layers
OutputLayers = [
    % Layers connect the seperate branches
    depthConcatenationLayer(2,'Name','joinTable');
    % reduce dimension to 256
    fullyConnectedLayer(256,'Name','GlobalFc');
    batchNormalizationLayer('Name','OutNorm');
    reluLayer('Name','Outrelu');
    % Drop  out the output with defauly probability 0.5
    dropoutLayer('Name','Outdrop');
    % Regularize the output the the formal responses for regression
    fullyConnectedLayer(29,'Name','OutputFc');
    OutputRegressionLayer('RegressionOutput',Options);
];

FullNet = addLayers(FullNet,OutputLayers);
FullNet = connectLayers(FullNet,'BodyFc','joinTable/in1');
FullNet = connectLayers(FullNet,'ImageFc','joinTable/in2');

end

% The Bodybranch
function BoLayers = Body_Layers
BoLayers = [
    FilterLayer('Bbranch','Body',224,128);
    % 1st part
    convolution2dLayer([1 3],32,'Stride',[1 2],'Padding',[0 1],'Name','Bconv1_1');
    reluLayer('Name','BReLu1_1');
    convolution2dLayer([3 1],64,'Stride',[2 1],'Padding',[1 0],'Name','Bconv1_2');
    batchNormalizationLayer('Name','BNorm1');
    reluLayer('Name','BReLu1_2');
    
    % 2nd part
    convolution2dLayer([1 3],128,'Stride',[1 2],'Padding',[0 1],'Name','Bconv2_1');
    reluLayer('Name','BReLu2_1');
    convolution2dLayer([3 1],128,'Stride',[2 1],'Padding',[1 0],'Name','Bconv2_2');
    % pay attention to the ReLU layer & BN layer here
    reluLayer('Name','BReLu2_2');
    batchNormalizationLayer('Name','BNorm2');
    
    % 3rd part
    convolution2dLayer([1 3],128,'Stride',[1 2],'Padding',[0 1],'Name','Bconv3_1');
    reluLayer('Name','BReLu3_1');
    convolution2dLayer([3 1],128,'Stride',[2 1],'Padding',[1 0],'Name','Bconv3_2');
    batchNormalizationLayer('Name','BNorm3');
    reluLayer('Name','BReLu3_2');
    
    averagePooling2dLayer([3 3],'Stride',[16 16],'Name','Bpool');
    
    % Output part
    fullyConnectedLayer(128,'Name','BodyFc');
];
end

% The Layers for Image Train 
function ImLayers = Image_Layers
ImLayers = [
    FilterLayer('Ibranch','Image',224,224);
    % 1st part
    convolution2dLayer([1 11],32,'Stride',[1 4],'Padding',[0 2],'Name','Iconv1_1');
    reluLayer('Name','IReLu1_1');
    convolution2dLayer([11 1],64,'Stride',[4 1],'Padding',[2 0],'Name','Iconv1_2');
    batchNormalizationLayer('Name','INorm1');
    reluLayer('Name','IReLu1_2');
    
    % 2nd part
    convolution2dLayer([1 5],128,'Stride',[1 2],'Padding',[0 2],'Name','Iconv2_1');
    reluLayer('Name','IReLu2_1');
    convolution2dLayer([5 1],256,'Stride',[2 1],'Padding',[2 0],'Name','Iconv2_2');
    % pay attention to the ReLU layer & IN layer here
    batchNormalizationLayer('Name','INorm2');
    reluLayer('Name','IReLu2_2');

    % 3rd part
    convolution2dLayer([1 3],384,'Stride',[1 2],'Padding',[0 1],'Name','Iconv3_1');
    reluLayer('Name','IReLu3_1');
    convolution2dLayer([3 1],512,'Stride',[2 1],'Padding',[1 0],'Name','Iconv3_2');
    batchNormalizationLayer('Name','INorm3');
    reluLayer('Name','IReLu3_2');
    
    % 4th part
    convolution2dLayer([1 3],384,'Stride',[1 1],'Padding',[0 1],'Name','Iconv4_1');
    reluLayer('Name','IReLu4_1');
    convolution2dLayer([3 1],384,'Stride',[1 1],'Padding',[1 0],'Name','Iconv4_2');
    batchNormalizationLayer('Name','INorm4');
    reluLayer('Name','IReLu4_2');
    
    % 5th part
    convolution2dLayer([1 3],640,'Stride',[1 2],'Padding',[0 1],'Name','Iconv5_1');
    reluLayer('Name','IReLu5_1');
    convolution2dLayer([3 1],640,'Stride',[2 1],'Padding',[1 0],'Name','Iconv5_2');
    batchNormalizationLayer('Name','INorm5');
    reluLayer('Name','IReLu5_2');
    
    % 6th part
    convolution2dLayer([1 3],640,'Stride',[1 1],'Padding',[0 1],'Name','Iconv6_1');
    reluLayer('Name','IReLu6_1');
    convolution2dLayer([3 1],640,'Stride',[1 1],'Padding',[1 0],'Name','Iconv6_2');
    batchNormalizationLayer('Name','INorm6');
    reluLayer('Name','IReLu6_2');
    
    % 7th part
    convolution2dLayer([1 3],640,'Stride',[1 2],'Padding',[0 1],'Name','Iconv7_1');
    reluLayer('Name','IReLu7_1');
    convolution2dLayer([3 1],640,'Stride',[2 1],'Padding',[1 0],'Name','Iconv7_2');
    batchNormalizationLayer('Name','INorm7');
    reluLayer('Name','IReLu7_2');
    
    % 8th part
    convolution2dLayer([1 3],640,'Stride',[1 1],'Padding',[0 1],'Name','Iconv8_1');
    reluLayer('Name','IReLu8_1');
    convolution2dLayer([3 1],640,'Stride',[1 1],'Padding',[1 0],'Name','Iconv8_2');
    batchNormalizationLayer('Name','INorm8');
    reluLayer('Name','IReLu8_2');
    
    averagePooling2dLayer([4 4],'Stride',[1 1],'Name','Ipool');
    % Output part
    fullyConnectedLayer(640,'Name','ImageFc');
];
end

