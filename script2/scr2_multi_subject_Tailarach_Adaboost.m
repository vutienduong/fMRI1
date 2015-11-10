clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
examples = [];
labels = [];
num_subjects = 6;

% create data (training + test)
for j=1:num_subjects
    clearvars -except j file_name examples labels num_subjects;
    load(file_name{j});
    % setting (2) normalize all trials
    % [info,data,meta] = transformIDM_normalizeTrials(info,data,meta);

    trials=find([info.cond]>1);
    [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);

    % setting (1) normalize image
    % [info1,data1,meta1] = transformIDM_normalizeImages(info1,data1,meta1); 

    % setting (3) normalize trials with condition 2,3
    [info1,data1,meta1] = transformIDM_normalizeTrials(info1,data1,meta1);

    [info1,data1,meta1] = createColToROI(info1,data1,meta1);
    [info2,data2,meta2] = transformIDM_avgROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});

    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='S'));

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

    examples=[examples; examplesP; examplesS];
    labels=[labels; labelsP; labelsS];
    disp(['Complete create part ', num2str(j)]);
end

% classify
c1 = cvpartition(labels,'k',num_subjects);
for i=1:num_subjects
    % tridx = c1.training(i);
    % teidx = c1.test(i);

    teidx = zeros(size(labels,1),1);
    teidx(((i-1)*80 + 1):(i*80), :) = 1;

    tridx = ones(size(labels,1),1) - teidx;

    teidx = logical(teidx);
    tridx = logical(tridx);

    extrain{1,i} = examples(tridx,:);
    labelstrain{1,i} = labels(tridx,:);
    extest{1,i} = examples(teidx,:);
    labelstest{1,i} = labels(teidx,:);

    model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',10,'Tree');
    testclass=predict(model, extest{1,i});

    corrects = sum(testclass == labelstest{1,i});
    acc(1,i) = corrects/length(labelstest{1,i});
    disp(['Complete train ADABOOST', num2str(i)]);
end
disp(['Accuracy', num2str(mean(acc))]);