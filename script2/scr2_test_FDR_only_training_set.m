% for each person , length(file_name)
clear; 
file_name = {'data-starplus-04799-v7', 'data-starplus-04847-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];
for j=1:6 %2:2:4
    clearvars -except j file_name all_acc;
    t = cputime;
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);

    % collect the non-noise and non-fixation trials
    trials=find([info.cond]>1); 
    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);

    % seperate P1st and S1st trials
    [info1,data1,meta1]=transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
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

    

    c1 = cvpartition(labelsP,'k',10);
    adb_acc = [];
    num_t_test = 10;

    % use Bayes or Ada here
    % 1)nbayes(1,2)   2)adaM1(1,2)     3)adaboost(1,-1) 4) knn(1,2) 
    % 5)svm (1,-1) 6)kernel perceptron(1,-1) 7) nn(1,-1): neuronnetwork 
    
    classifier = 'nbayes';

    for i=1:num_t_test
        tridx = c1.training(i);
        teidx = c1.test(i);

        examplesP_train = examplesP(tridx,:);
        examplesS_train = examplesS(tridx,:);

        % Fisher ===========================
        numfeat = size(examplesP_train,2);
        for ii=1:numfeat
            fdr(ii)= Fisher(examplesP_train(:,ii),examplesS_train(:,ii));
        end
        [fdr,featrank]=sort(fdr,'descend');
        % end Fisher ===========================

        % choosing 100 features based on FDR values
        nf = 100;
        examplesPR = examplesP(:,featrank); 
        examplesSR = examplesS(:,featrank);
        examplesPS = examplesPR(:,1:nf); 
        examplesSS = examplesSR(:,1:nf);

        % create train set with 100 features based on examplesPS & examplesSS
        extrain{1,i} = [examplesPS(tridx,:); examplesSS(tridx,:)];
        labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];

        % create test set with 100 features based on examplesPS & examplesSS
        extest{1,i} = [examplesPS(teidx,:); examplesSS(teidx,:)];
        labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

        % classify
        if strcmp(classifier,'adaM1')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'adaM1');
        elseif strcmp(classifier,'nbayes')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'nbayes');
        elseif strcmp(classifier,'adaboost')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'adaboost');
        elseif strcmp(classifier,'knn')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'knn');
            disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
        elseif strcmp(classifier,'svm')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'svm');
            disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
        elseif strcmp(classifier,'perce')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'perce');
            disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
        elseif strcmp(classifier,'nn')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'nn');
            disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
        end
    end
    avg_acc = sum(adb_acc)/num_t_test;
    e = cputime - t;
    t = cputime;
    all_acc = [all_acc; avg_acc e];
    disp(['accuracy ', num2str(avg_acc) , ' | processing time ', num2str(e)]);
end
mean_all = mean(all_acc);
disp(['AVERAGE ACC ', num2str(mean_all(1))]);
disp(['AVERAGE PROCESSING TIME ', num2str(mean_all(2))]);


% ROis, 50, AdaM1(50): 86.25% 34.54s [0.8625;0.9625;0.9250;0.725;0.85;0.85]
% ROis, 100, AdaM1(50): 87.29% 36.21s [0.8625;0.9750;0.8875;0.725;0.90;0.8875]
% ROis, 100, adaboost(3000): 89.58% 31.55s [0.9;0.9750;0.9375;0.775;0.8875;0.9]
% ROis, 50, adaboost(3000): 89.17% 31.48s [0.9;1;0.925;0.7125;0.9375;0.875]
% ROis, 20, adaboost(3000): 87.29% 31.51s [0.875;0.975;0.9375;0.7250;0.8625;0.8625]
% ROis, 30, adaboost(3000): 89.17% 31.45s [0.9125;1;0.9125;0.7375;0.9125;0.8750]
% 100, adaboost(3000): 90.21% 73.38s [0.8875;1;0.95;0.8125;0.8750;0.8875]


% 30. knn(3) 86.875% 36.21s [0.8625;0.9750;0.925;0.6625;0.9;0.8875]
% 50. knn(7) 89.79% 32.18s [0.8875;0.9875;0.9375;0.7875;0.925;0.8625]

% 50. svm(linear, C=0.5, tol=0.001: 88.75%   40.3887 s [0.8875;0.9875;0.925;0.7125;0.9125;0.90]
% 50. svm(poly(1,3), C=2, tol=0.001: 85.417%   33.4726 s [0.8;0.95;0.875;0.725;0.9125;0.8625]
% 50. svm(poly(1,3), C=0.5, tol=0.001: 86.042%   33.4726 s [0.85;0.9625;0.8875;0.725;0.91250;0.825]

% 50. perce(linear, C=2, tol=0.001: 0.75625   32.2688 s [0.85;0.9625;0.8875;0.725;0.91250;0.825]decompose_tensor_Tucker
% 50. perce(rbf(1), C=0.5, tol=0.001: 0.83542   32.2714
% 50. perce(rbf(1.5), C=0.5, tol=0.001:  0.84583
% 50. perce(rbf(0.1), C=0.5, tol=0.001: 100%   32.2324 s
% 50. perce(poly(1,3), C=0.5, tol=0.001: 0.76667   32.4664 s 

