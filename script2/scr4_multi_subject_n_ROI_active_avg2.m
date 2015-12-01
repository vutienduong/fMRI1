% this script use to test: 
% first, choose N most active voxels for each ROI(use ActiveVoxact), 
% then calculate Average of this N voxels, represent for a ROI
clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
examples = [];
labels = [];
num_subjects = 6;
N = 20;
use_avg_nFDR = true;
create_data = true;

if create_data
    % create data (training + test)
    for j=1:num_subjects
        clearvars -except j file_name examples labels num_subjects N use_avg_nFDR;
        % best ROIs
        ROIs = {'CALC' 'LDLPFC' 'LIPL' 'LIPS' 'LOPER'  'LT' 'LTRIA'};

        load(file_name{j});

        % normalize (1) a voxel through all time points in a trial
        [info,data,meta] = transformIDM_normalizeTrials2(info,data,meta);
        [info,data,meta] = transformIDM_normalizeImages(info,data,meta);

        % 1. Avg(n Active)/ROI
        %[examplesP, examplesS] = avg_nActivePerROICondFixed(info,data,meta, ROIs, N);

        % 2. [mean std kurtoris skewness]/ROI
        [examplesP, examplesS] = meanStdKurtSkewCond(info,data,meta, ROIs);

        labelsP=ones(size(examplesP,1),1);
        labelsS=ones(size(examplesS,1),1)+1;
        examples=[examples; examplesP; examplesS];
        labels=[labels; labelsP; labelsS];
        disp(['Complete create part ', num2str(j)]);
    end
end

% classify
for i=1:num_subjects
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
        testclass=predict(model,extest{1,i});
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
disp(['Std ', num2str(std(acc))]);


% ======================= RESULTS ===============================
% norm trial: 76
% norm trial + img: 78
% norm img: 63.75