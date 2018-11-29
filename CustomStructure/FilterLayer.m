% FilterLayer('Name',...,'Part',...)
% FilterLayer should be put behind the imageInputLayer.
% This layer is desined to complete the model EMOTIC, CVPR 2017 in MATLAB
% 2018b.
% Since multiple input layers are not supported in MATLAB 2018b,
% this layer is designed to filter the input images and send 
% them to different branches. Meanwhile, images will be Normalized in this Layer.
%
% The Template is from ImageInput.m( Deep learning toolbox.)
% Yue-Kai, USTC.
%
classdef FilterLayer < nnet.layer.Layer
    properties       
        % Part, a parameter decides wheter this layer should pick body part 
        % or original images
        Part
        InputWidth
        OutputWidth
        Resize
    end

    properties (Constant)
        % (Optional) Layer learnable parameters.
        %
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
        % Global Parameters to Normalization the input images.
        Global_mean =  single([0.45897533113801; ...
                               0.44155118600299; 0.40552199274783]);
        Global_std = single( [0.23027497714954;...
                                0.22752317402935;0.23638979553161]);
        % Layer learnable parameters go here.
    end
    
    methods
        function self = FilterLayer(name,part,inputWidth,outputWidth)
            % (Optional) Create a myLayer.
            % This function must have the same name as the layer.
            % Layer constructor function goes here.
            self.Name = name;
            self.Part = part;
            self.InputWidth = inputWidth;
            self.OutputWidth = outputWidth;
            self.Resize = (inputWidth~=outputWidth);
        end
        
        function Z = predict(self, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         X        -    Input data, a 4D augmented matrix
            % Output:
            %         Z        -    Output of layer forward function
            
            % Layer forward function for prediction goes here.
            % The original images are augmented with zeros
            % and cropped bodies are augmented with ones
            if strcmp(self.Part,'Body')
                Z = X(:,:,1:3,:);
            elseif strcmp(self.Part,'Image')
                Z = X(:,:,4:6,:);
            end
            if (self.Resize) && ~isempty(Z)
                Z = imresize(Z,[self.OutputWidth,self.OutputWidth]);
            end
        end

        function [Z,memory] = forward(self,X)
            % Same as the predict
            if strcmp(self.Part,'Body')
                Z = X(:,:,1:3,:);
            elseif strcmp(self.Part,'Image')
                Z = X(:,:,4:6,:);
            end
            if (self.Resize) && ~isempty(Z)
                Z = imresize(Z,[self.OutputWidth,self.OutputWidth]);
            end
            memory = [];
        end

        function dLdX = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            %
            dLdX = 0*X;
            % This Layer does not need backward function
        end
    end
end