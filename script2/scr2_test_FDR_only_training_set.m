% for each person , length(file_name)
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
    labelsP=ones(size(examplesP,1),1);
    labelsS=ones(size(examplesS,1),1)+1;

    examples=[examplesP;examplesS];
    labels=[labelsP;labelsS];

    

    c1 = cvpartition(labelsP,'k',10);
    adb_acc = [];
    num_t_test = 10;

    % use Bayes or Ada here
    use_ada = false;

    for i=1:num_t_test
        tridx = c1.training(i);
        teidx = c1.test(i);

        examplesP_train = examplesP(tridx,:);
        examplesS_train = examplesS(tridx,:);

        % dung ra la phai tinh Fisher o day, vi chon chi dua vao training set, k dc xet toan bo training + test set
        % co the thu phuong an validation set

        % Fisher ===========================
        numfeat = size(examplesP_train,2);
        for ii=1:numfeat
            fdr(ii)= Fisher(examplesP_train(:,ii),examplesS_train(:,ii));
        end
        [fdr,featrank]=sort(fdr,'descend');
        % end Fisher ===========================

        % choosing 100 features based on FDR values
        examplesPR = examplesP(:,featrank); 
        examplesSR = examplesS(:,featrank);
        examplesPS = examplesPR(:,1:100); 
        examplesSS = examplesSR(:,1:100);

        % create train set with 100 features based on examplesPS & examplesSS
        extrain{1,i} = [examplesPS(tridx,:); examplesSS(tridx,:)];
        labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];

        % create test set with 100 features based on examplesPS & examplesSS
        extest{1,i} = [examplesPS(teidx,:); examplesSS(teidx,:)];
        labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

        %==============ADABOOST======================
        if use_ada
            model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',120,'Tree');
            testclass=predict(model,extest{1,i});
            corrects = sum(testclass == labelstest{1,i});
            adb_acc(i) = corrects/length(labelstest{1,i});
            % disp(['pause ...']);
            % pause
        else
            % Bayes
            [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
            [predictions] = applyClassifier(extest{1,i},classifier);
            [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
            adb_acc(i) = 1- result{1,1};
            % disp(['temp accuracy ', num2str(adb_acc(i))]);
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