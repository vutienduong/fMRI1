% this script use to test: 
% after generating examples, calculate [mean std kurtoris skewness] of a ROI 
clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
examples = [];
labels = [];
num_subjects = 6;
N = 20;
use_avg_nFDR = true;
create_data = false;
multisubjs_classify = false;

if create_data
    % create data (training + test)
    exampleS1 = [];
    exampleP1 = [];
    for j=1:num_subjects
        clearvars -except j file_name examples num_subjects N use_avg_nFDR exampleS1 exampleP1 multisubjs_classify;
        % best ROIs
        ROIs = {'CALC' 'LDLPFC' 'LIPL' 'LIPS' 'LOPER'  'LT' 'LTRIA'};

        load(file_name{j});

        % normalize (1) a voxel through all time points in a trial
        [info,data,meta] = transformIDM_normalizeTrials2(info,data,meta);
        [info,data,meta] = transformIDM_normalizeImages(info,data,meta);

        % 2. [mean std kurtoris skewness]/ROI
        %[examplesP, examplesS] = meanStdKurtSkewCond(info,data,meta, ROIs, 'after_generate_examples'); % no bias
        [examplesP, examplesS] = meanStdKurtSkewCond(info,data,meta, ROIs, 'after_generate_examples', 'bias'); % bias
        %[examplesP, examplesS] = meanStdKurtSkewCond(info,data,meta, ROIs); % voxels
        
        exampleS1 = [exampleS1; examplesS];
        exampleP1 = [exampleP1 ;examplesP];
        
        disp(['Complete create part ', num2str(j)]);
    end
    
    % normalization
    norm_method = 2;
    
    if norm_method == 1 % 1. std
        [c1,c2]=normalizeStd( exampleS1', exampleP1');
    elseif norm_method == 2  % 2. [-1, 1]
        [c1,c2]=normalizeMnmx (exampleS1', exampleP1', -1 , 1 );
    else % 3. [0, 1]
        [c1,c2]=normalizeSoftmax (exampleS1', exampleP1', 0.5 );
    end
    
    exampleS1 = c1';
    exampleP1 = c2';
    save('scr54index.mat','exampleS1','exampleP1');
    
else
    load('scr54index.mat');
end

labelsS = ones(size(exampleS1,1),1);
labelsP = ones(size(exampleP1,1),1) + 1;
if multisubjs_classify % multi subjects   
    for i=1:num_subjects
        teidx = zeros(size(exampleS1,1),1);
        teidx(((i-1)*40 + 1):(i*40), :) = 1;
        tridx = not(teidx);
        teidx = logical(teidx);
        tridx = logical(tridx);

        extrain{1,i} = [exampleS1(tridx,:); exampleP1(tridx,:)];
        labelstrain{1,i} = [labelsS(tridx,:); labelsP(tridx,:)];
        extest{1,i} = [exampleS1(teidx,:); exampleP1(teidx,:)];
        labelstest{1,i} = [labelsS(teidx,:); labelsP(teidx,:)];

        % Bayes
        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        acc(1,i) = 1- result{1,1};
    end
    disp(['Accuracy', num2str(mean(acc))]);
    disp(['Std ', num2str(std(acc))]);
    
else % single subject
    all_subj_acc = [];
    for i=1:num_subjects
        all_acc = [];
        tindex = ((i-1)*40 + 1):(i*40);
        dataP = exampleP1( tindex, :);
        dataS = exampleS1( tindex, :);
        labelP = labelsP( tindex);
        labelS = labelsS( tindex);
        
        c1 = cvpartition(labelsS(tindex),'k',10);
        for i2=1:10
            tridx = c1.training(i2);
            teidx = c1.test(i2);

            % FDR 2
%             numSelectedFeat = 10;
%             extrainP = dataP(tridx,:);
%             extrainS = dataS(tridx,:);
%             numfeat = size(extrainP,2);
%             for ii=1:numfeat
%                 fdr(ii)= Fisher(extrainP(:,ii),extrainS(:,ii));
%             end
%             [fdr,featrank]=sort(fdr,'descend');
% 
%             selectedIndex = featrank(1:numSelectedFeat);
%             examplesPR = dataP(:,selectedIndex); 
%             examplesSR = dataS(:,selectedIndex);
            % end FDR 2
            examplesPR = dataP;
            examplesSR = dataS;

            extrain{1,i2} = [examplesPR(tridx,:);examplesSR(tridx,:)];
            labelstrain{1,i2} = [labelP(tridx,:);labelS(tridx,:)];
            extest{1,i2} = [examplesPR(teidx,:);examplesSR(teidx,:)];
            labelstest{1,i2} = [labelP(teidx,:);labelS(teidx,:)];

            [classifier] = trainClassifier(extrain{1,i2},labelstrain{1,i2},'nbayes');
            [predictions] = applyClassifier(extest{1,i2},classifier);
            [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i2});
            all_acc(i2) = 1- result{1,1};
        end
        all_subj_acc = [all_subj_acc, mean(all_acc(:))];
    end
    disp(['Accuracy', num2str(mean(all_subj_acc))]);
    disp(['Std ', num2str(std(all_subj_acc))]);
end


% classify multi subject

% ======================= RESULTS ===============================
% dang examples(no BIAS) : norm trial + img: 75% + 9  (norm hay k norm, hay norm bang pp 1,2 hay 3 cung k doi)
% dang voxels:   norm trial + img: 78% + 12 (norm hay k norm, hay norm bang pp 1,2 hay 3 cung k doi)