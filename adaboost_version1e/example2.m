clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04820-v7', 'data-starplus-04847-v7',...
    'data-starplus-05675-v7', 'data-starplus-05680-v7', 'data-starplus-05710-v7'};

total_result = [];
list_RT = {'RT'};
% for each person , length(file_name)
for j=2:2
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);
    % 1)=============================
    % [in,d,m]=transformIDM_selectTrials(info,data,meta,find([info.cond]~=0));
    % [info1, data1, meta1] = transformIDM_avgROIVoxels(in,d,m, list_RT);
    % [examples,labels,expInfo] = idmToExamples_condLabel(info1, data1, meta1);
    % END 1)=========================

    % 2)=============================
    % collect the non-noise and non-fixation trials
    trials=find([info.cond]>1); 
    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
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

    % END 2)=========================
    
    %train and test itself
    clsf = 'nbayes';
    [classifier] = trainClassifier(examples,labels,clsf);   %train classifier
    [predictions] = applyClassifier(examples,classifier);       %test it
    [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labels);
    disp(['   Accuracy for single person ', int2str(j)]);
    acc = 1 - result{1};
    total_result = [total_result acc];
    disp(acc);

    %==============ADABOOST======================
    datafeatures=[];
    dataclass = [];


    % for k=1:length(labels)
    %   if(labels(k) ~= 3)
    %     datafeatures = [datafeatures; examples(k,:)];
    %     if(labels(k) == 2)
    %       dataclass = [dataclass; -1];
    %     else
    %       dataclass = [dataclass; labels(k)];
    %     end
    %   end
    % end

    datafeatures = examples(:, 100:250);
    dataclass = labels;
    dataclass(dataclass == 2) = -1;

    % Use Adaboost to make a classifier
    [classestimate,model]=adaboost('train',datafeatures,dataclass,20);

    % Show the error verus number of weak classifiers
    error=zeros(1,length(model)); for i=1:length(model), error(i)=model(i).error; end 
    subplot(2,2,3), plot(error); title('Classification error versus number of weak classifiers');

    % Make some test data
    testdata = datafeatures;

    % Classify the testdata with the trained model
    testclass=adaboost('apply',testdata,model);

    % Show result
    corrects = sum(testclass == dataclass);
    adb_acc = corrects/length(dataclass);

    % Show the data
    disp(['AVERAGE FOR SINGLE PERSON BY adaboost']);
    disp(adb_acc);

    %==============End ADABOOST======================

end
disp(['AVERAGE FOR SINGLE PERSON BY nbayes']);
disp(mean(total_result));



