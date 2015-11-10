% for each person , length(file_name)
% this file use FDR, correlation feature, exhauste search and nbayes
clear; 
file_name = {'data-starplus-04799-v7', 'data-starplus-04847-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];
for j=1:6
    clearvars -except j file_name all_acc;
    t = cputime;
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);
    
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
    labelsS=ones(size(examplesS,1),1)+1;

    % Fisher ===========================
    numfeat = size(examplesP,2);

    for i=1:numfeat
        fdr(i)= Fisher(examplesP(:,i),examplesS(:,i));
    end
    
    [fdr,featrank]=sort(fdr,'descend');
    
    % Transport
    T =[fdr',featrank'];
    T = T(1:100,:); % TEST
    c1_train = examplesP';
    c2_train = examplesS';
    disp(['DONE Transport !', num2str(cputime-t)]);
    
    % 2. The scalar feature ranking technique which employs FDR
    % in conjunction with a feature correlation measure.
    [p]= compositeFeaturesRanking (c1_train,c2_train,0.2,0.8,T);
    % TEST
    disp(['DONE compositeFeaturesRanking !', num2str(cputime-t)]);

    % 3. In order to reduce the dimensionality of the feature space, work with the 7 highest ranked features
    inds=sort(p(1:15),'ascend');
    c1_train=c1_train(inds,:);
    c2_train=c2_train(inds,:);

    % 4. Choose the best feature combination consisting of three features (out of the previously selected seven), using the exhaustive search method.
    [cLbest,Jmax]= exhaustiveSearch(c1_train,c2_train,'ScatterMatrices',[10]);
    % TEST
    disp(['DONE exhaustiveSearch !', num2str(cputime-t)]);

    % 5. Form the resulting training dataset (using the best feature combination), along with the corresponding class labels.
    examplesPS = c1_train(cLbest,:)'; 
    examplesSS = c2_train(cLbest,:)';

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

        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        adb_acc(1,i) = 1- result{1,1};
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

% result
% 100 features, 10, 3: 89.58%   9.9s
% 200 features, 10, 3: 91.04%   14.06s  

% 50  features, 20, 5: 94.17%   11.88s
% 100 features, 20, 5: 94.79%   12.50s 

% 50  features, 20, 7: 96.88%   26.24s [0.9625;1;0.975;0.975;0.9625;0.9375]
% 100  features, 15, 10: 98.13%   10.01s [0.9750;1;0.975;0.975;0.9875;0.9750]
%                        98.33%   9.89s [0.9875;1;0.975;0.975;0.9875;0.9750]