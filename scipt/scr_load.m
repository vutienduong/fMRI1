clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04820-v7', 'data-starplus-04847-v7',...
    'data-starplus-05675-v7', 'data-starplus-05680-v7', 'data-starplus-05710-v7'};
for i=1:3%length(file_name)
    load(file_name{i});
    disp(['Complete load : ', file_name{i}]);
    [in,d,m]=transformIDM_selectTrials(info,data,meta,find([info.cond]~=0));
    [info1, data1, meta1] = transformIDM_avgROIVoxels(in,d,m, {'RT'});
    [examples1,labels1,expInfo1] = idmToExamples_condLabel(info1, data1, meta1);
    
    if ~exist('examples', 'var') && ~exist('labels', 'var') && ~exist('expInfo', 'var')
        examples = examples1;
        labels = labels1;
        expInfo = expInfo1;
    else
        %{
        disp('   Size examples    : '); 
        disp(size(examples));
        disp('   Size labels    : ');
        disp(size(labels));
        disp('   Size expInfo    : ');
        disp(size(expInfo));
        
        disp(['   --------------']);
        disp('   Size examples(1)    : '); 
        disp(size(examples1));
        disp('   Size labels(1)    : ');
        disp(size(labels1));
        disp('   Size expInfo(1)    : ');
        disp(size(expInfo1));
        %}
        
        %disp(['   Size examples(1) : ', size(examples1)]);
        %disp(['   Size labels(1)   : ', size(labels1)]);
        %disp(['   Size expInfo(1)  : ', size(expInfo1)]);
        
        [examples, labels, expInfo] = mergeExamples(examples, labels, expInfo, examples1, labels1, expInfo1);    
    end
    disp(['   Complete merge : ', int2str(i)]);
end

[classifier] = trainClassifier(examples,labels,'nbayes');   %train classifier
[predictions] = applyClassifier(examples,classifier,'nbayes');       %test it
[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labels);
disp(['   Accuracy for test set idenify with training set ']);
1 - result{1}

% use person 05710-v7 to test
test_index = length(file_name);
load(file_name{i});
[in,d,m]=transformIDM_selectTrials(info,data,meta,find([info.cond]~=0));
[info1, data1, meta1] = transformIDM_avgROIVoxels(in,d,m, {'RT'});
[examplesTest,labelsTest,expInfoTest] = idmToExamples_condLabel(info1, data1, meta1);

[predictionsTest] = applyClassifier(examplesTest,classifier,'nbayes');      
[resultTest,predictedLabels,traceTest] = summarizePredictions(predictionsTest,classifier,'averageRank',labelsTest);

disp(['   Accuracy for test set is  ID 05710 ']);
1 - resultTest{1}
