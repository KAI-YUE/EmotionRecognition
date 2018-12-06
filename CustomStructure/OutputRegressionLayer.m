% This is a Regression Layer to output both discrete and continuous 
% value of EMOTIC Model. 
% The loss function is defined as follows:
% Lcomb = lamda_disc * L_disc+ lamda_cont * Lcont
% where,
% L_disc = 1/N*sum(wi*(y'-y).^2), wi = 1/log(c+P_i)
% Lcont = 1/#c*sum(vk*(y'-y).^2), vk belongs to {0,1}
% when |y'-y|<threshold theta, vk =0
% lamda weights the importance of each loss
% 
% ( Input/output annotations should be a Nx29 matrix, where 1:26 denotes 
% the discrete dimension and 27:29 denotes continuous VAD dimension.)
% All of these parameters are set in structure Parameters by defalut, in
% which case wi equals to each other, lamda = [1/6 1] and theta equals 
% to 0.5;
% Make a change if necessary.
% Yue-Kai, USTC.

classdef OutputRegressionLayer < nnet.layer.RegressionLayer
    properties
      % Default Layer properties.
      Parameters = struct('theta',0.5,'lamda',[1/6;1],'PrThreshold',0.7,'Bias_c',1.2);
      NumCategories
      NumContinuous
    end
 
    methods
        function self = OutputRegressionLayer(name,parameters)           
            % (Optional) Create a Img_RegressionLayer.
            self.Name = name;
            self.Description = 'Sum of Error';
            
            categories = {'Peace'; 'Affection'; 'Esteem'; 'Anticipation'; 'Engagement'; 
            'Confidence'; 'Happiness'; 'Pleasure'; 'Excitement'; 'Surprise'; 'Sympathy'; 
             'Doubt/Confusion'; 'Disconnection'; 'Fatigue'; 'Embarrassment';
            'Yearning'; 'Disapproval'; 'Aversion'; 'Annoyance'; 'Anger'; 'Sensitivity';
             'Sadness';  'Disquietment'; 'Fear'; 'Pain'; 'Suffering'};
            self.NumCategories = length(categories);
            self.NumContinuous = 3;
            self.ResponseNames = cat(1,categories,{'Valence';'Arousal';'Dominance'});
            
            tags = fieldnames(parameters);
            tags_len = length(tags);
            if(length(tags)~=length(fieldnames(self.Parameters)))
                error('OutputRegressionLayer: Unknownoption');
            end
            for i=1:tags_len
                if isfield(self.Parameters,tags{i})
                    self.Parameters.(tags{i}) = parameters.(tags{i}); 
                end
            end
        end

        function L = forwardLoss(self, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T.
            %
            % Inputs:
            %         self - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T
            Ydisc = Y(:,:,1:self.NumCategories,:);
            Ycont = Y(:,:,self.NumCategories+1:end,:);
            Tdisc = T(:,:,1:self.NumCategories,:);
            Tcont = T(:,:,self.NumCategories+1:end,:);
            
            % Set the weight
            vk = (abs(Ycont - Tcont) > self.Parameters.theta);
            TotalCategories = sum(Tdisc,'all');
            Pr = sum(Tdisc,4)/TotalCategories;
            Wi = 1./log(Pr+self.Parameters.Bias_c);
            
            lamda_disc = self.Parameters.lamda(1);
            lamda_cont = self.Parameters.lamda(2);
            % In order to avoid division, we take 1/26 = 0.038462
            % and 1/3 = 0.333333
            Error = lamda_disc*sum(Wi.*(Ydisc-Tdisc).^2,3)*0.038462 + ...
                    lamda_cont*sum(vk.*(Ycont-Tcont).^2,3)*0.333333;
            
            % Take mean value over batchsize
            N = size(Y,4);
            L = sum(Error)/N;
        end
        
        function dLdY = backwardLoss(self, Y, T)
            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         self - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y
            
            Ydisc = Y(:,:,1:self.NumCategories,:);
            Ycont = Y(:,:,self.NumCategories+1:end,:);
            Tdisc = T(:,:,1:self.NumCategories,:);
            Tcont = T(:,:,self.NumCategories+1:end,:);
            
            % Set the weight, same as forwardloss
            vk = (abs(Ycont - Tcont) > self.Parameters.theta);
            TotalCategories = sum(Tdisc,'all');
            Pr = sum(Tdisc,4)/TotalCategories;
            Wi = 1./log(Pr+self.Parameters.Bias_c);
            
            lamda_disc = self.Parameters.lamda(1);
            lamda_cont = self.Parameters.lamda(2);
            % Due to the derivative, we take 1/13 = 0.076923 and 2/3 =
            % 0.666667
            dLdY = cat(3,lamda_disc*Wi.*(Ydisc-Tdisc)*0.076923,...
                   lamda_cont*vk.*(Ycont - Tcont)*0.666667);
            
            % Mask the gradients
            mask = (abs(dLdY) > 1e-9);
            dLdY = mask.*dLdY;
        end
    end
end
