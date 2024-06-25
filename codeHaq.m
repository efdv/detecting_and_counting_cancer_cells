% clc
% clear all
% close all

% dd

% input size.
imageSize = [224 224 3];

%no of class
numClasses = 1;

anchorBoxes = [
    43 59
    18 22
    23 29
    84 109
];


base = resnet50;

inputlayer=base.Layers(1)

middle =base.Layers(2:174)

finallayer=base.Layers(175:end)

baseNetwork=[inputlayer
    
               middle
               
               finallayer]

% Specify the feature extraction layer.

featureLayer = 'activation_48_relu';

%% the YOLO v2 object detection network. 

lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,base, featureLayer);

options = trainingOptions('adam', ...
        'MiniBatchSize',4, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',200,'Plots','training-progress');    
% % % %     
%vehicleDataset=gTruth;
vehicleDataset=trainingData;

[detector,info] = trainYOLOv2ObjectDetector(vehicleDataset,lgraph,options);

% testing
aa=imread('2.jpg')

figure,imshow(aa)

[bboxes,scores,labels] = detect(detector,aa);

II = insertObjectAnnotation(aa,'rectangle',bboxes,labels);

    figure
    pause(0.5)
    imshow(II)
    title('testing')
    
% %     counting

c1=sum(labels=='impurities ')

c2=sum(labels=='HT29')





imds1 = imageDatastore('C:\Users\asifr\Desktop\inyat\CCC-HT-29 dataset by Prof\20200907\20200907',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

for i = 1:5
%     subplot(7,7,i)
    I = readimage(imds1,i);
    [bboxes,scores,labels] = detect(detector,I);
    
II = insertObjectAnnotation(I,'rectangle',bboxes,labels);

    figure
    pause(0.5)
    imshow(II)
    title('testing')
end

 
imds1 = imageDatastore('C:\Users\JITECJESUS\Downloads\archive (14)\data\training_images',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');


gt=outt(1:200,2);

blds = boxLabelDatastore(gt);


[testdata] = splitEachLabel(imds1,200);

for i=1:200
    
    aa=readimage(imds1,i);
    
    imds1.Files(i)
    
    [bboxes,scores,label] = detect(detector,aa);

  I = insertObjectAnnotation(aa,'rectangle',bboxes,scores);
%   
%   figure(1)
%   imshow(I)
  
  pause(1)
end



results = detect(detector, testdata);


[ap, recall, precision] = evaluateDetectionPrecision(results, blds);

figure;
plot(recall, precision,'Linewidth',2);
xlabel('Precision')
ylabel('Recall')
legend('Yolo')
grid on


         
