clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04820-v7', 'data-starplus-04847-v7',...
    'data-starplus-05675-v7', 'data-starplus-05680-v7', 'data-starplus-05710-v7'};
totalResult = [];
totalError  = [];

roiActiveAvg_condition = 'null';
useRoiActiveAvg_condition = 0 ;
useNActive_condition = 1;
useNRoiActive_condition = 0;
nSelect = 15;
method = 'nbayes';

for i=1:length(file_name)
    load(file_name{i});
    disp(['Complete load : ', file_name{i}]);
    
    %-------------CREATE EXAMPLE--------------------
    %roiActiveAvg
    if strcmp(roiActiveAvg_condition, 'null')
        allRois = {meta.rois.name};
        roiActiveAvg_condition = allRois(:);
    end
    
    
    if useRoiActiveAvg_condition
        trials = find([info.cond]>1); 
        [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
        [info1, data1, meta1] = transformIDM_avgROIVoxels(info1,data1,meta1,roiActiveAvg_condition);
    %NActive chon dung 20
    elseif useNActive_condition
        trials = find([info.cond]>0); 
        [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
        [info1, data1, meta1] = transformIDM_selectActiveVoxact(info1,data1,meta1,nSelect, [2, 3]);
    
    %NRoiActive, co the len den 28
    elseif useNRoiActive_condition
        trials = find([info.cond]>0); 
        [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
        [info1, data1, meta1] = transformIDM_selectActiveVoxels(info1,data1,meta1,nSelect);
    else
        trials = find([info.cond]>1); 
        [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);     
    end
    
    
    %trials=find([info.cond]>0); 
    %[info1,data1,meta1]=transformIDM_selectTrials(info1,data1,meta1,trials);
    
    
        % seperate P1st and S1st trials
    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));
 
        % seperate reading P vs S
    [infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
    [infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
    [infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
    [infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);

        % convert to examples
    [examplesP2,labelsP2,exInfoP2]=idmToExamples_condLabel(infoP2,dataP2,metaP2);
    [examplesP3,labelsP3,exInfoP3]=idmToExamples_condLabel(infoP3,dataP3,metaP3);
    [examplesS2,labelsS2,exInfoS2]=idmToExamples_condLabel(infoS2,dataS2,metaS2);
    [examplesS3,labelsS3,exInfoS3]=idmToExamples_condLabel(infoS3,dataS3,metaS3);

        % combine examples and create labels.  Label 'picture' 1, label 'sentence' 2.
    examplesP=[examplesP2;examplesP3];
    examplesS=[examplesS2;examplesS3];
    labelsP=ones(size(examplesP,1),1);
    labelsS=ones(size(examplesS,1),1)+1;
    examples=[examplesP;examplesS];
    labels=[labelsP;labelsS];
    
    %{
    if useNRoiActive_condition || useNActive_condition
        [classifier] = trainClassifier(examples,labels,'nbayes');   %train classifier
        [predictions] = applyClassifier(examples,classifier);       %test it
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'error',labels);
    else
    %}
    
    max_size = size(examples,1);
    for j=1:max_size
        trainIndex = [1:j-1 j+1 : max_size ];
        examplesTrain = examples(trainIndex, :);
        labelsTrain = labels(trainIndex, :);

        examplesTest = examples(j, :);
        labelsTest = labels(j, :);

        % train a Naive Bayes classifier
        [classifier] = trainClassifier(examplesTrain,labelsTrain,method);   %train classifier
        [predictions] = applyClassifier(examplesTest,classifier);       %test it
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'error',labelsTest);
        totalResult(j) = predictedLabels == labelsTest;
    end
    errorArr = totalResult(totalResult == 0);
    totalError(i) = size(errorArr,2) / max_size;
    totalResult = [];
        
    % summarize the results of the above predictions.   
     %
     %totalResult(i) = 1-result{1};  % rank accuracy
    %-------------end CREATE EXAMPLE--------------------
end
averageError = mean(totalError)
