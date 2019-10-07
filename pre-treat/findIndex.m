% Input a file name, this function finds the index of it in the annotation.
% It can locates the flawed data in the original annotations.
%

function [idx,order] = findIndex(annotations,fileName)
found = 0;
dirRoot = 'D:\Big_Data\emotic';
order = regexp(fileName,['_person\d*','.'],'match');
order = order{1};
order = regexp(order,['\d*'],'match');
order = order{1};
for i =1:length(annotations)
    fullFileName = fullfile(dirRoot,annotations(i).folder,annotations(i).filename);
    personString = ['_person',order,'.jpg'];
    fullFileName = strrep(fullFileName,'.jpg',personString);
    fullFileName = strrep(fullFileName,'images','bodies');
    if strcmp(fullFileName,fileName)
        found =1;
        break;
    end
end
if ~found
    idx =[];
    order = [];
else
    idx =i;
    order = str2num(order);
end