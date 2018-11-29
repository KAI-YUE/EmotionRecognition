% ds = bindImageDatastore(outputsize,tbl,augment)
% input,
% outputsize: 1x2 array equals to the inputsize of the Image
% tbl:        table variable stores the filepath and responses of cropped
%             bodies
% augment:    structer determines whether a random horizoontal filp or a 
%             small angle rotation will be applied to the images.
%
% output,
% ds, a data structure compatible to the ImageInputLayer.
% 
% This class is defined to bind the orignal images and the cropped bodies 
% together. BindImages have six planes, and they will be seperated in the 
% filterLayer.It is designed for the EMOTIC model in CVPR2017.
% The template is from MATLAB augmentedImageDatastore.
% Yue- Kai,USTC.
% 
classdef bindImageDatastore < matlab.io.Datastore & ...
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
        AugmentOptions = struct('rotate',false,'flip',false');
        Augmentation = false;
    end

    properties(Access = private)
        % This property is inherited from Datastore
        CurrentFileIndex
        
        % These properties are imitated from augmentedImageDatastore
        % DatastoreInternal
        OutputSize
        
    end


    methods
        
        function self = bindImageDatastore(outputsize,tbl,augmentOptions)
            % Construct function
            self.OutputSize = outputsize;
            self.NumObservations = height(tbl);
            self.MiniBatchSize = 1;
            self.CurrentFileIndex = 1;
            self.Files = tbl{:,1};
            self.Responses = tbl{:,2:end};
            self.NumResponses = size(self.Responses,2);
            % DispatchInBackground is set to false by default.
            self.DispatchInBackground = false;
            % Set Augment options
            if exist('augment','var')
                tags = fieldnames(self.AugmentOptions);
                    for i=1:length(tags)
                        if isfield(augmentOptions,tags{i})
                            self.AugmentOptions.(tags{i})=logical(augmentOptions.(tags{i}));
                            if augmentOptions.(tags{i}), self.Augmentation = true; end
                        end
                    end
            end
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
            bindImages = cell(self.MiniBatchSize,1);
            responses = cell(self.MiniBatchSize,1);
            for i = 1:self.MiniBatchSize
                idx = self.CurrentFileIndex+i-1;
                % Read Body regions and then resize it.
                Body = imread(self.Files{idx});
                Body = imresize(Body,self.OutputSize);
                % Read the original Image and then resize it.
                ImageFilePath = strrep(self.Files{idx},'bodies','images');
                PersonIndex = regexp(self.Files{idx},'_person\d*','match');
                ImageFilePath = strrep(ImageFilePath,PersonIndex{1},'');
                Image = imread(ImageFilePath);
                Image = imresize(Image,self.OutputSize);
                % Bind the Body and Image
                bindImages{i} = cat(3,Body,Image);
                responses{i} = reshape(self.Responses(idx,:),1,1,self.NumResponses);
            end
            self.CurrentFileIndex = self.CurrentFileIndex + i;
            
            if self.Augmentation
                bindImages = self.augmentationToBatch(bindImages);
            end
            % Arrange the data into table.
            data = table(bindImages,responses);
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
            Xout = cell(self.MiniBatchSize,1);
            if self.AugmentOptions.rotate
                for i = 1:self.MiniBatchSize
                    randAngle = randi([-3 3],1);
                    Xout{i} = imrotate(X{i},randAngle,'nearest','crop');
                end
            end
            if self.AugmentOptions.flip
                for i = 1:self.MiniBatchSize
                    if randi([0 1])
                        Xout{i} = X{i}(:,end:-1:1,:);
                    end
                end
            end
        end
        
    end
end % end class definition
