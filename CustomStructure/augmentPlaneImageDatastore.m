% ds = augmentPlaneImageDatastore(outputsize,tbl,randflip)
% input,
% outputsize: 1x2 array equals to the inputsize of the Image
% tbl:        table variable stores the filepath and responses
% randfli:    logical variable,False by deault. If set to True, a horizontal 
% flip transformation will be applied to the image when it is read randomly.
%
% output,
% ds, a data structure compatible to the ImageInputLayer.
% 
% This class is defined to add characteristic matrix to the output of
% ImageInputLayer, hence the dataset can be separated in the following
% layers. It is designed for the EMOTIC model in CVPR2017.
% Specifically, if the image is original, then a zero matrix will be 
% caught to the third diminsion of the matrix. Instead, if the image
% is cropped body, then a eyes matrix will be caught to the third
% dimension.
% The template is from MATLAB augmentedImageDatastore.
%
% 
classdef augmentPlaneImageDatastore < matlab.io.Datastore & ...
                       matlab.io.datastore.MiniBatchable &...
                       matlab.io.datastore.Shuffleable &...
                       matlab.io.datastore.PartitionableByIndex&...
                       matlab.io.datastore.BackgroundDispatchable 
    
    properties
        MiniBatchSize
    end
    
    properties(SetAccess = protected)
        NumObservations
        
        % Custom properties
        Files
        Responses
        NumResponses
        Randflip = false;
    end

    properties(Access = private)
        % This property is inherited from Datastore
        CurrentFileIndex
        
        % These properties are imitated from augmentedImageDatastore
        % DatastoreInternal
        OutputSize
        
    end


    methods
        
        function self = augmentPlaneImageDatastore(outputsize,tbl,randflip)
            % Construct function
            self.OutputSize = outputsize;
            self.NumObservations = height(tbl);
            self.MiniBatchSize = 1;
            self.CurrentFileIndex = 1;
            self.Files = table2array(tbl(:,1));
            self.Responses = table2array(tbl(:,2:end));
            self.NumResponses = size(self.Responses,2);
            if exist('randflip','var'), self.Randflip = logical(randflip);   end
        end

        function tf = hasdata(self)
            % Return true if more data is available
            tf = self.CurrentFileIndex <= self.NumObservations;
        end

        function [data,info] = read(self)            
            % Read one batch of data
            if ~self.hasdata()
               error('AugmentedPlaneImageDatastore outOfData'); 
            end
            images = cell(self.MiniBatchSize,1);
            responses = cell(self.MiniBatchSize,1);
            for i = 1:self.MiniBatchSize
                idx = self.CurrentFileIndex+i-1;
                images{i} = imread(self.Files{idx});
                responses{i} = reshape(self.Responses(idx,:),1,1,self.NumResponses);
            end
            self.CurrentFileIndex = self.CurrentFileIndex + i;
            
            % Arrange the data into table.
            images = self.augmentationToBatch(images);
            data = table(images,responses);
            info.CurrenReadIndices = self.CurrentFileIndex;
        end
        
        % Define background dispatch method
        function [data,info] = readByIndex(self,indices)
            indices_len = length(indices);
            images = cell(indices_len,1);
            responses = cell(indices_len,1);
            for i = 1:indices_len
                idx = indices(i);
                images{i} = imread(self.Files{idx});
                responses{i} = reshape(self.Responses(idx,:),1,1,self.NumResponses);
            end
            images = self.augmentationToBatch(images);
            data = table(images, responses);
            info.CurrenReadIndices = indices;
        end

        function reset(self)
            % Reset to the start of the data
            self.CurrentFileIndex = 1;
        end
        
        function set.MiniBatchSize(self,batchSize)
            self.MiniBatchSize = batchSize;
        end
        
        function batchSize = get.MiniBatchSize(self)
            batchSize = self.MiniBatchSize;
        end
        
        % Define shuffable method
        function dsnew = shuffle(self)
            dsnew = copy(self);
            shuffledIndexOrder = randperm(self.NumObservations);
            dsnew.Files = self.Files(shuffledIndexOrder);
            dsnew.Responses = self.Responses(shuffledIndexOrder,:);
        end
        
        % Define parallel or multi-GPU method
        function dsnew = partitionByIndex(self,indices)  
           dsnew = copy(self);
           dsnew.Files = self.Files(indices);
           dsnew.CurrentFileIndex = 1;
           dsnew.Responses = self.Responses(indices,:);
           dsnew.NumObservations = length(dsnew.Files);           
        end
        
        function outputSize = getOutputSize(self)
            outputSize = self.OutputSize;
        end
        
    end
        
    methods (Hidden = true)

        function frac = progress(self)
            % Determine percentage of data read from datastore
            frac = (self.CurrentFileIndex-1)/self.NumObservations;
        end

    end
    
    methods (Access = private)
        
        function Xout = augmentationToBatch(self,X)
            batchSize = length(X);
            Xout = cell(batchSize,1);
            for obs = 1:batchSize
                X{obs} = self.resizeData(X{obs});
                % Judge if the image is original or cropped body
                % if it is original, add characteristic matrix 1 to the 4th
                % dimension; else add matrix 0 to 4th dimension
                h = self.OutputSize(1);
                w = self.OutputSize(2);
                if isempty(strfind(self.Files{obs},'person'))
                    Xout{obs} = cat(3,X{obs},zeros(h,w));
                else           
                    Xout{obs} = cat(3,X{obs},ones(h,w));
                end
            end
            if self.Randflip
                for obs = 1:batchSize
                    if randi([0 1]), Xout{obs} = Xout{obs}(:,end:-1:1,:); end
                end
            end
        end
        
         function Xout = resizeData(self,X)
            inputSize = size(X);
            if isequal(inputSize(1:2),self.OutputSize)
                Xout = X; % no-op if X is already desired OutputSize
                return
            else
                Xout = imresize(X, self.OutputSize, 'method','bilinear');
            end
         end
         
    end
end % end class definition
