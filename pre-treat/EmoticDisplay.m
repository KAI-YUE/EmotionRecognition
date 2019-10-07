function EmoticDisplay(annotations,Mode,order)
% This function displays an image from EMOTIC database
% with categories and VAD histogram.
% input,
% annotation: a structure from the EMOTIC database
% type: 1,    annotation is from 'train' 
%       2,    annotation is from 'val'/'test'
% order:      which person should be stressed in the image
% It only displays one image once. Change the DiskRoot if necessary.

headcount = length(annotations(1).person);
if ~exist('order','var')
    order = 1; 
elseif order > headcount
    order = headcount;
end
if ~exist('Mode','var'),    Mode = 1;   end

if Mode == 1
    VAD =  annotations(1).person(order).annotations_continuous;
    Categories = annotations(1).person(order).annotations_categories.categories;
else
    VAD = annotations(1).person(order).combined_continuous;
    Categories = annotations(1).person(order).combined_categories;
end

DiskRoot = 'D:\Big_Data\emotic';
foldername = fullfile(DiskRoot,annotations(1).folder,annotations(1).filename);
image = imread(foldername);

bbox = annotations(1).person(order).body_bbox;
width = bbox(3) - bbox(1)+1;
height = bbox(4) - bbox(2)+1;


subplot(4,4,[1:12]),
imshow(image);
rectangle('Position',[bbox(1:2),width,height],'cur',0.1,'EdgeColor','r','LineWidth',0.75);

axis off;
annotation('textbox',[0.18 0.08 0.2 0.2],'String',Categories,'Color',[65 105 225]/255,...
                'LineStyle','none','FontSize',16);
hold on;

subplot(4,4,[15:16]),
histogram('Categories',{'V';'A';'D'},'BinCounts'...
           ,[VAD.valence,VAD.arousal,VAD.dominance],'BarWidth',0.5);
ylim([0 10]);


end




