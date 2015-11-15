% AFFIRMATIVE, NEGATIVE
% bayes, adaboost, adaM1
clear; 
file_name = {'data-starplus-04799-v7', 'data-starplus-04847-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];
for j=1:6
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
    
    % load label AFF (1), NEG (-1) from "aff_neg_data2"
    [labelsPS2, labelsPS3] = aff_neg_data2();
    labels   = [labelsPS2'; labelsPS3'; labelsPS2'; labelsPS3'];
    examples = [examplesP; examplesS];
    affidx = labels==1;
    
    % examplesP now is affirmative, examplesS now is negative
    examplesP = examples(affidx,:);
    examplesS = examples(not(affidx),:);
    
    % labelsP now is affirmative(1), labelsS now is negative(-1)
    labelsP=ones(size(examplesP,1),1);
    labelsS=ones(size(examplesS,1),1)-2;

    c1 = cvpartition(labelsP,'k',10);
    adb_acc = [];
    num_t_test = 10;

    % use Bayes or Ada here
    % 1)nbayes   2)adaM1     3)adaboost
    classifier = 'adaboost';

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
            model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',50,'Tree');
            testclass=predict(model,extest{1,i});
            corrects = sum(testclass == labelstest{1,i});
            acc = corrects/length(labelstest{1,i});
            disp(['Train adaM1 ']);
        elseif strcmp(classifier,'nbayes')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'nbayes');
        elseif strcmp(classifier,'adaboost')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'adaboost');
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


% ROis, 100, Bayes: 57.97% 30.97s [0.8625;0.9625;0.9250;0.725;0.85;0.85]
% ROis, 100, AdaM1(50): fail
% ROis, 100, adaboost(3000): 57.71% 34.06s [0.55;0.4875;0.675;0.5375;0.5375;0.675]]
% ROis, 30, adaboost(3000):


% TODO: thuc hien lai vs tung ROIs, sau do chon cac ROIs tot nhat