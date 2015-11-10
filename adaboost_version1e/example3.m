% for each person , length(file_name)
clear; 
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
for j=1:6
    clearvars -except j file_name;
    t = cputime;
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);
    % 1)=============================
    % [in,d,m]=transformIDM_selectTrials(info,data,meta,find([info.cond]~=0));
    % [info1, data1, meta1] = transformIDM_avgROIVoxels(in,d,m, list_RT);
    % [examples,labels,expInfo] = idmToExamples_condLabel(info1, data1, meta1);
    % END 1)=========================

    % 2)=============================
    % collect the non-noise and non-fixation trials
    trials=find([info.cond]>1); 
    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
    % seperate P1st and S1st trials
    [info2,data2,meta2]=transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
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
    labelsS=ones(size(examplesS,1),1)-2;

    % Fisher ===========================
    numfeat = size(examplesP,2);
    % so sanh tung feature, moi feature tinh cho tat ca row cua examples
    for i=1:numfeat
        fdr(i)= Fisher(examplesP(:,i),examplesS(:,i));
    end

    examples=[examplesP;examplesS];
    labels=[labelsP;labelsS];

    [fdr,featrank]=sort(fdr,'descend');
    examplesPR = examplesP(:,featrank); 
    examplesSR = examplesS(:,featrank);
    examplesPS = examplesPR(:,1:70); 
    examplesSS = examplesSR(:,1:70);

    c1 = cvpartition(labelsP,'k',10);
    adb_acc = [];
    num_t_test = 10;
    for i=1:num_t_test
        tridx = c1.training(i);
        teidx = c1.test(i);
        extrain{1,i} = [examplesPS(tridx,:);examplesSS(tridx,:)];
        labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];
        extest{1,i} = [examplesPS(teidx,:);examplesSS(teidx,:)];
        labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

        %==============ADABOOST======================
        % Use Adaboost to make a classifier
        itt = 40; % number of weak classifiers
        [classestimate,model]=adaboost('train', extrain{1,i}, labelstrain{1,i}, itt);

        % Show the error verus number of weak classifiers

        % error=zeros(1, length(model)); for ii=1:length(model), error(ii)=model(ii).error; end 
        % subplot(2,2,3), plot(error); title('Classification error versus number of weak classifiers');

        % Make some test data
        testdata = extest{1,i};

        % Classify the testdata with the trained model
        testclass=adaboost('apply',testdata,model);

        % Show result
        corrects = sum(testclass == labelstest{1,i});
        adb_acc(i) = corrects/length(labelstest{1,i});
        % disp(['finish']);
        % disp(i);
        %==============End ADABOOST======================
    end
    % end Fisher ===========================
    disp(['AVERAGE ACCURACY', num2str(sum(adb_acc)/num_t_test)]);
    e = cputime - t;
    t = cputime;
    disp(['Processing time is ', num2str(e)]);
    % END 2)=========================
end




