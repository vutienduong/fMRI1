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
    % 1)=============================04847
    % [in,d,m]=transformIDM_selectTrials(info,data,meta,find([info.cond]~=0));
    % [info1, data1, meta1] = transformIDM_avgROIVoxels(in,d,m, list_RT);
    % [examples,labels,expInfo] = idmToExamples_condLabel(info1, data1, meta1);
    % END 1)=========================

    % 2)=============================
    % collect the non-noise and non-fixation trials
    trials=find([info.cond]>1); 
    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
    % seperate P1st and S1st trials
    [info1,data1,meta1]=transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
    [info1,data1,meta1] = transformIDM_selectActiveVoxact(info1,data1,meta1,240, [2 3]);
    % trials=find([info1.cond]>1); 
    % [info1,data1,meta1]=transformIDM_selectTrials(info1,data1,meta1,trials);
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
    for i=1:num_t_test
        use_ada = false;
        tridx = c1.training(i);
        teidx = c1.test(i);
        extrain{1,i} = [examplesP(tridx,:);examplesS(tridx,:)];
        labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];
        extest{1,i} = [examplesP(teidx,:);examplesS(teidx,:)];
        labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

        %==============ADABOOST======================

        if use_ada
            % Ada
            model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',50,'Tree');
            testdata = extest{1,i};
            testclass=predict(model,testdata);
            corrects = sum(testclass == labelstest{1,i});
            adb_acc(i) = corrects/length(labelstest{1,i});
        else
            % Bayes
            [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
            [predictions] = applyClassifier(extest{1,i},classifier);
            [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
            adb_acc(i) = 1- result{1,1};
        end
    end
    % end Fisher ===========================
    avg_acc = sum(adb_acc)/num_t_test;
    e = cputime - t;
    t = cputime;
    all_acc = [all_acc; avg_acc e];
    disp(['accuracy ', num2str(avg_acc) , ' | processing time ', num2str(e)]);
    % END 2)=========================
end
mean_all = mean(all_acc);
disp(['AVERAGE ACC', num2str(mean_all(1))]);
disp(['AVERAGE PROCESSING TIME ', num2str(mean_all(2))]);

% (1) cond>0 => ROIs => 100 mostActive (127) => cond>1 
% (2) cond>1 => ROIs => 100 mostActiveVoxact

% (1) + Bayes => 80.208 (53.75, 93.75, 91.25, 68.75, 88.75, 85.00)
% (1) + Ada   => 82.50  (56.25, 97.50, 88.75, 72.50, 91.25, 88.75) ~30s

% (2) + Bayes => 80 (100), 83 (240), 85 (500)