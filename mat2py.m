clc
clear all

% data = load('../image labeler/Labels/labels.mat');
data = load('../image labeler/SegPCLabels/data.mat');
path = data.gTruth.DataSource.Source;
matriz = data.gTruth.LabelData;
[u,v] = size(matriz);
cont = 0;
for i=1:u
    I = imread(strcat(path{i}));
    fprintf("%s \n", strcat(path{i}))
    for j=1:v
        if j == 1
            %rsave = '..\dataset\crops\impurities\';
            rsave = '..\..\..\datasets\TCIA_SegPC_dataset\crops\imp\';
        %elseif j == 2
         %   rsave = '...\dataset\groundTrue\impurities\'; 
        end
        data = load('../image labeler/SegPCLabels/data.mat');
        matriz1 = matriz(i,j);
        matriz1 = table2cell(matriz1);
        for m=1:length(matriz1{1})
            crop = imcrop(I, matriz1{1}(m,:));
            imwrite(crop, strcat(rsave, int2str(cont), '.png'))
            cont = cont + 1;
        end
        % general_coo = data.gTruth.LabelData.cell{1};
        % general_coo = data.gTruth.LabelData.impurity{1};
        general_coo = data.gTruth.LabelData.imputiries{1};
    end
end




%%
%rutaDirectorio = '..\dataset\crops\ht-29\';
%rutaDirectorio = '..\dataset\crops\impurities\';
%rootSave = '..\dataset\crops\ht-29_20x20\';
%rootSave = '..\dataset\crops\impurities_20x20\';
rutaDirectorio = '..\..\..\datasets\TCIA_SegPC_dataset\crops\imp\';
rootSave = '..\dataset\crops\impurities2\';
contenidoDirectorio = dir(rutaDirectorio);
for i=3:length(contenidoDirectorio)-2
    I = imread(strcat(rutaDirectorio, contenidoDirectorio(i,1).name));
    [u,v,c] = size(I);
    fprintf("%s\n",contenidoDirectorio(i,1).name);
    if u > 20 && v > 20
        newI = imresize(I, [30,30]);
        imwrite(newI, strcat(rootSave, contenidoDirectorio(i,1).name))
    end
end

%% get counted cells

data = load('../image labeler/Couting/data.mat');
path = data.gTruth.DataSource.Source;
matriz = data.gTruth.LabelData;
[u,v] = size(matriz);

cont = 0;
for i=1:u
    for j=1:v
        matriz1 = matriz(i,j);
        matriz1 = table2cell(matriz1);
        for m=1:length(matriz1{1})
            crop = imcrop(I, matriz1{1}(m,:));
            imwrite(crop, strcat(rsave, int2str(cont), '.png'))
            cont = cont + 1;
        end
        % general_coo = data.gTruth.LabelData.cell{1};
        general_coo = data.gTruth.LabelData.impurity{1};
    end
end

%%
trainingData = objectDetectorTrainingData(gTruth);
imageFileNames = trainingData.imageFilename;
bboxLabels = trainingData(:, 2:end);
imds = imageDatastore(imageFileNames);
blds = boxLabelDatastore(bboxLabels);
trainingData = combine(imds, blds);

disp(trainingData);

