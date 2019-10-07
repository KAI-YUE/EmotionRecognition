% sigmoidLayer('Name')
% This layer is desined to complete the model EMOTIC, CVPR 2017 in MATLAB
% 2018b. Since it is a muti-label task, we leverages sigmoid function to
% the fullyconnectesLayer.
%
% The Template is from ImageInput.m( Deep learning toolbox.)
%
%
classdef sigmoidLayer < nnet.layer.Layer
    properties (Access = protected)      
        % This Layer dose not need any learnable parameters.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty();
    end

    methods
        function self = sigmoidLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the layer.
            % Layer constructor function goes here.
            self.Name = name;           
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         X        -    Input data 
            % Output:
            %         Z        -    Output of layer forward function
            Z = 1/(1+exp(-X));
        end

        function dLdX = backward(~, X, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            % Derivative of sigmoid function: exp(-x)./(1+exp(-x)).^2;
            dLdX = exp(-X)./(1+exp(-X)).^2 .* dLdZ;
        end
    end
end