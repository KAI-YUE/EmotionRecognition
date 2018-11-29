% This function splits the bodies from the EMOTIC database and returns
% the table stores the folderName and responses .The original images and 
% cropped bodies are put together in the data table. For each full image,
% we take the union of the categories and mean VAD of all the people in it 
% as its responses.
% 
% input,
% annotations: belongs to the set {train, test, val};
% Mode:       
%       1, cropped bodies won't be rewrite. 
%       2, rewrite all of the cropped bodies.
% set to 2 by default.
% Mode 1 is only feasible if the annot_tbl is deleted by accident while all
% of the cropped bodies have been generated before.
%
% output,
% a table contains:
% folderNameArray, array storing the folderName of bodies (type: 'string')
% responseArray, a lx29 array catch the multi-labels (type: 'logical') and
% VAD (type:'double')
%
% Example:
% TestBTbl = splitBodies(test,2);
%
% Change the RootDir if necessary.
% Yue-Kai, USTC.
% (Attention: for cropped bodies we only take the 1st annotation, in order to come across
% contradictary categories.)

function annot_tbl = splitBodies(annotations,Mode)

RootDir = 'D:\Big_Data\emotic';
database = {'ade20k';'emodb_small';'framesdb';'mscoco'};
labels = {'Peace'; 'Affection'; 'Esteem'; 'Anticipation'; 'Engagement'; 
'Confidence'; 'Happiness'; 'Pleasure'; 'Excitement'; 'Surprise'; 'Sympathy'; 
'Doubt/Confusion'; 'Disconnection'; 'Fatigue'; 'Embarrassment';
'Yearning'; 'Disapproval'; 'Aversion'; 'Annoyance'; 'Anger'; 'Sensitivity';
'Sadness';  'Disquietment'; 'Fear'; 'Pain'; 'Suffering'};
% Respones are used as names of columns when the final table is generated
% '\' can not be appeared in the variableNames, hence it is changed to '_'
Responses = cat(1,labels,{'Valence';'Arousal';'Dominance'});
Responses{12} = 'Doubt_Confusion';
labels_len = length(labels);
images_num = length(annotations);

% Total count of people
headcount = 0;
for i=1:images_num
    headcount = headcount+length(annotations(i).person);
end

% Initialize the array and parameters.
folderNameArray = cell(2*headcount,1);
ResponsesMatrix =  zeros(2*headcount,labels_len+3,'single');
index = 0;

if ~exist('Mode','var'),    Mode = 2;   end
if Mode ==2 
%-----------------------Default Mode------------------------
% mkdir the bodies files in Mode 2
    database_len = length(database);
    for i = 1:database_len
        sub_folder = fullfile(RootDir,database{i},'bodies');
        if ~isfolder(sub_folder)
            mkdir (sub_folder);
        end
    end

    for i=1:images_num
        ImagePath = fullfile(RootDir,annotations(i).folder,annotations(i).filename);
        I = imread(ImagePath);

        num_people = length(annotations(i).person);
        save_people = strrep(ImagePath,'images','bodies');
        
        tempLabels = zeros(1,labels_len,'logical');
        tempVAD = zeros(1,3);
        for j =1:num_people
            
            index = index+1;
            folderNameArray{index} = ImagePath;
            index = index+1;
            % bbox has the format [x1 y1 x2 y2]
            % x refers to the horizontal coordinate
            % y refers to the vertical coordinate
            bbox = annotations(i).person(j).body_bbox;
            width = bbox(3) - bbox(1)+1;
            height = bbox(4) - bbox(2)+1;
            person = imcrop(I,[bbox(1:2),width,height]);

            % Save the body images to ...\bodies\
            subfolder = sprintf('%s%d%s','_person',j,'.jpg');
            folderNameArray{index} =  strrep(save_people,'.jpg',subfolder);

            if ~isa(person,'uint8'),    person = im2double(person); end
            imwrite(person,folderNameArray{index},'BitDepth',8);

            % Add responses to responseArray
            contin_vad = annotations(i).person(j).annotations_continuous(1);
            idx = ismember(labels,annotations(i).person(j).annotations_categories(1).categories);
            ResponsesMatrix(index,idx) = 1;
            ResponsesMatrix(index,labels_len+1) = contin_vad.valence;
            ResponsesMatrix(index,labels_len+2) = contin_vad.arousal;
            ResponsesMatrix(index,labels_len+3) = contin_vad.dominance;
            
            tempLabels = bitor(tempLabels,logical(ResponsesMatrix(index,1:labels_len)));
            tempVAD = tempVAD + ResponsesMatrix(index,labels_len+1:end);
        end
        
        idx = index-2*j+1:2:index-1;
        ResponsesMatrix(idx,1:labels_len) = repmat(tempLabels,length(idx),1);
        ResponsesMatrix(idx,labels_len+1:end) = repmat(tempVAD./num_people,length(idx),1);
    end
    
%-----------------------------Mode 1-------------------------------------
else
    for i=1:images_num
    ImagePath = fullfile(RootDir,annotations(i).folder,annotations(i).filename);
    save_people = strrep(ImagePath,'images','bodies');
    num_people = length(annotations(i).person);

    tempLabels = zeros(1,labels_len,'logical');
    tempVAD = zeros(1,3);
        for j =1:num_people
            
            index = index+1;
            folderNameArray{index} = ImagePath;
            index = index+1;
            
            subfolder = sprintf('%s%d%s','_person',j,'.jpg');
            folderNameArray{index} =  strrep(save_people,'.jpg',subfolder);
            
            % Add responses to responseArray
            contin_vad = annotations(i).person(j).annotations_continuous(1);
            idx = ismember(labels,annotations(i).person(j).annotations_categories(1).categories);
            ResponsesMatrix(index,idx) = 1;
            ResponsesMatrix(index,labels_len+1) = contin_vad.valence;
            ResponsesMatrix(index,labels_len+2) = contin_vad.arousal;
            ResponsesMatrix(index,labels_len+3) = contin_vad.dominance;
            
            tempLabels = bitor(tempLabels,logical(ResponsesMatrix(index,1:labels_len)));
            tempVAD = tempVAD + ResponsesMatrix(index,labels_len+1:end);
        end
        idx = index-2*j+1:2:index-1;
        ResponsesMatrix(idx,1:labels_len) = repmat(tempLabels,length(idx),1);
        ResponsesMatrix(idx,labels_len+1:end) = repmat(tempVAD./num_people,length(idx),1);
    end
end  

response_tbl = array2table(ResponsesMatrix,'VariableNames',Responses);
folderName_tbl = table(folderNameArray,'VariableNames',{'folderName'});
annot_tbl = cat(2,folderName_tbl,response_tbl);
end
