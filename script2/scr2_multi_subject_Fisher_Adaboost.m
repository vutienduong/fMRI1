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

    [info,data,meta] = transformIDM_normalizeTrials2(info,data,meta);
    trials=find([info.cond]>1);
    [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);
    [info1,data1,meta1] = transformIDM_normalizeImages(info1,data1,meta1); 
    [info2,data2,meta2] = transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
    [info2,data2,meta2] = transformIDM_selectActiveVoxact(info2,data2,meta2,300, [2 3]);

    % trials=find([info2.cond]>1);
    % [info2,data2,meta2] = transformIDM_selectTrials(info2,data2,meta2,trials);
    use_FDR = false;


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

    % Fisher ===========================
    % setting 1, FDR dc tinh cho tung subject
    if use_FDR
        numfeat = size(examplesP,2);

        for i=1:numfeat
            fdr(i)= Fisher(examplesP(:,i),examplesS(:,i));
        end

        [fdr,featrank]=sort(fdr,'descend');
        examplesPR = examplesP(:,featrank); 
        examplesSR = examplesS(:,featrank);
        examplesPS = examplesPR(:,1:100); 
        examplesSS = examplesSR(:,1:100);

        % end Fisher

        examples=[examples; examplesPS; examplesSS];
    else
        examples=[examples; examplesP; examplesS];
    end

    labels=[labels; labelsP; labelsS];

    disp(['Complete create part ', num2str(j)]);
end

% classify
c1 = cvpartition(labels,'k',num_subjects);
for i=1:num_subjects
    % tridx = c1.training(i);
    % teidx = c1.test(i);
    use_ada = false;

    teidx = zeros(size(labels,1),1);
    teidx(((i-1)*80 + 1):(i*80), :) = 1;

    tridx = ones(size(labels,1),1) - teidx;

    teidx = logical(teidx);
    tridx = logical(tridx);

    extrain{1,i} = examples(tridx,:);
    labelstrain{1,i} = labels(tridx,:);
    extest{1,i} = examples(teidx,:);    
    labelstest{1,i} = labels(teidx,:);

    if use_ada
        % Ada
        model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',120,'Tree');
        testdata = extest{1,i};
        testclass=predict(model,testdata);
        corrects = sum(testclass == labelstest{1,i});
        acc(1,i) = corrects/length(labelstest{1,i});
        disp(['Complete train ADABOOST', num2str(i)]);
    else
        % Bayes
        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        acc(1,i) = 1- result{1,1};
    end
end
disp(['Accuracy', num2str(mean(acc))]);

% ==============    RESULTS     =============================
% 100 features, 100 ite == 77,70%
% 100 features, 120 ite == 77,70%
% 50  features, 100 ite == 76.25%
% 50  features, 120 ite == 76.25%
% 20  features, 100 ite == 77.08%
% 10  features, 100 ite == 76.88%

% ==============    IMPROVEMENTS     =============================
% 2) setting 2 gop tat ca S cua 6subject lai, tuong tu cho P. dung so voxel cua subject co so voxel lon nhat lam chuan
% neu subject nao thieu thi them 0 vao ->DONE
% 3) setting 3 dung Fisher cho tung ROI, chang han lay 20 feature tu moi ROI

% 100 features, 120 ite, normalize trails(before choosing ROIs) == 84.583%
% 100 features, 120 ite, normalize trails(after choosing ROIs) == 84.583%
% 100 features, 120 ite, normalize trails, /dev (before choosing ROIs) == 84.583%
% 100 features, 120 ite, normalize images == 77.083%
% 100 features, 120 ite, normalize images & trails(before choosing ROIs) == 90.208%
% 100 features, 120 ite, normalize images & trails(no choose ROIs) == 87.917%
% 100 features, 120 ite, norm(trials), 100 most active voxels == %
% 100 features, 120 ite, norm(img), 100 most active voxels == %
% 100 features, 120 ite, norm(trials + img), 100 most active voxact == 0.59792%
% 100 features, 120 ite, norm(trials + img), 100 most active voxact == 0.53958%

% 200 most active voxact [2]   == 0.71875%
% 200 most active voxact [2 3] == 0.71875%

% 300 most active voxact [2]   == 0.73125%
% 300 most active voxact [2 3] == 0.73125%

% high: 0 -> voxel -> 1
% high: 1 -> voxact