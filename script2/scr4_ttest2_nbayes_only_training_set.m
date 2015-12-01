% for each person , length(file_name)
% this file use ttest2 and nbayes, k chia
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
    
    examplesPS = examplesP;
    examplesSS = examplesS;

    c1 = cvpartition(labelsP,'k',10);
    adb_acc = [];
    num_t_test = 10;
    
    % 1)nbayes   2)adaM1     3)adaboost
    classifier = 'adaM1';
    
    for i=1:num_t_test
        tridx = c1.training(i);
        teidx = c1.test(i);
        extrain{1,i} = [examplesPS(tridx,:);examplesSS(tridx,:)];
        labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];
        extest{1,i} = [examplesPS(teidx,:);examplesSS(teidx,:)];
        labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];
        
        % ttest2
        rho = 0.0001;
        [h] = ttest2(examplesPS(tridx,:),examplesSS(tridx,:),rho);
        h = find(h);
        extrain{1,i} = extrain{1,i}(:,h);
        extest{1,i} = extest{1,i}(:,h);

        % classify
        if strcmp(classifier,'adaM1')
            adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'adaM1');
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
disp(['AVERAGE ACC', num2str(mean_all(1))]);
disp(['AVERAGE PROCESSING TIME ', num2str(mean_all(2))]);
    
% result: Bayes
% 1. k chia (rho = 0.01)
%   ROIs, rho = 0.0001  88.125% 2.496s [0.8875;0.9875;0.9125;0.7;0.925;0.8625]
%   ROIs, rho = 0.005   89.583% 2.27 [0.9375;0.9875;0.925;0.775;0.875;0.875]
%         rho = 0.0001  87.5%  4.9348s [0.8375;1;0.925;0.725;0.925;0.8375]
%         rho = 0.005   89.38%  5.0674 [0.875;0.9875;0.925;0.8;0.8875;0.8875]

% adaM1
% ROIs, adaM1(100) 86.04% 23.23 s [0.825;1;0.9375;0.6125;0.8875;0.9]  rho = 0.01
% ROIs, adaM1(100) 86.67% 19.11 s [0.775;0.9625;0.9375;0.725;0.9;0.9] rho = 0.005
% ROIs, adaM1(100) 84.79% 12.24 s [0.825;0.975;0.9;0.6125;0.875;0.9] rho = 0.0001
% ROIs, adaM1(200) 87.71% 22.06 s [0.875;0.9875;0.875;0.725;0.875;0.925] rho = 0.0001

