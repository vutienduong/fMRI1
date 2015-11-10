clear;
t = cputime;
file_name = {'data-starplus-04847-v7', 'data-starplus-04820-v7', 'data-starplus-04847-v7',...
    'data-starplus-05675-v7', 'data-starplus-05680-v7', 'data-starplus-05710-v7'};

list_RT = {'RT', 'CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC'};
% for each person , length(file_name)
for i=1:1
    load(file_name{i});
    disp(['Complete load : ', file_name{i}]);
    [in,d,m]=transformIDM_selectTrials(info,data,meta,find([info.cond]~=0));
    [info1, data1, meta1] = transformIDM_avgROIVoxels(in,d,m, list_RT);
    [examples,labels,expInfo] = idmToExamples_condLabel(info1, data1, meta1);
    
    %train and test itself
    % clsf = 'nbayes';
    % [classifier] = trainClassifier(examples,labels,clsf);   %train classifier
    % [predictions] = applyClassifier(examples,classifier,clsf);       %test it
    % [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labels);
    % disp(['   Accuracy for single person ', int2str(i)]);
    % acc = 1 - result{1};
    % total_result = [total_result acc];
    % disp(acc);

    c1 = cvpartition(labels,'k',10);
    for i=1:10
        tridx = c1.training(i);
        teidx = c1.test(i);
        extrain{1,i} = examples(tridx,:);
        labelstrain{1,i} = labels(tridx,:);
        extest{1,i} = examples(teidx,:);
        labelstest{1,i} = labels(teidx,:);

        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        acc(1,i) = result{1,1};
    end

    avacc = 1-sum(acc)/10;
    disp(['Average accuracy is ', num2str(avacc)]);
    e = cputime - t;
    disp(['Processing time is ', num2str(e)]);
end