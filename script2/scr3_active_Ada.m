% this file uses 5 setting
% 1:ActiveVoxels, 
% 2:ActiveVoxact, 
% 3:nActive/ROI 
% 4:avgROIVoxel 
% 5:avg(nActive/ROI)
% 6:nActive/ROI & mFDR
% 7:avgROIVoxel & mFDR

clear; 
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];
for j=1:6
    clearvars -except j file_name all_acc;
    
    % 1:ActiveVoxels, 2:ActiveVoxact, 3:nActive/ROI 4:avgROIVoxel 5:avg(nActive/ROI)
    load(file_name{j});
    use_setting = 1;
    num_selected_feature = 100; % use for 1,2
    num_per_ROI = 50; % use for 3
    ROIs = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
    t = cputime;
%     trials=find([info.cond]>0); 
%     [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);
    disp(['Complete load : ', file_name{j}]);
    
    
    if use_setting == 1 % ActiveVoxact
        % this cond is higher accuracy than (>0)
        % >0: 72% (240)
        % >1: 84% (240)
        % (>0 =>voxact => >1) 82%(240)
        [info1,data1,meta1] = activeVoxactCond(info,data,meta,ROIs,num_selected_feature);
        
    elseif use_setting == 2 % ActiveVoxels
        % (>1 =>activeVoxels) thi 6 sub [94 88 94 70 77 83] =>AVG:83.75%
        % (>0 =>activeVoxels) AVG:70%, sub2 la 46%(?)
        % (>0 =>activeVoxels => >1) 6 sub [98 55(?) 89 73 85 85] =>AVG:81%
        %                           neu k tinh sub2, thi AVG(5): 86%    
        [info1,data1,meta1] = activeVoxelsCond(info,data,meta,ROIs,num_selected_feature);
        
    elseif use_setting == 3 % nActive/ROI
        % (>0)  68% (Voxact = Voxels)
        % (>1)  61.5 %
        % (>0 =>nActive/ROI => >1) 78.5%, sub2(55%(?))
        [info1,data1,meta1] = nActivePerROICond(info,data,meta,ROIs,num_per_ROI);
    elseif use_setting == 4 % nActive/ROI
        % 71%
        [info1,data1,meta1] = avgROIVoxelCond(info,data,meta,ROIs);
    elseif use_setting == 5 % avg(nActive/ROI) (su ket hop cua 3 + 4)
        % 75%
        [info1,data1,meta1] = avg_nActivePerROICond(info,data,meta,ROIs,num_per_ROI);
    end
    
    
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
    examplesP=examplesP3;
    examplesS=examplesS2;
    labelsP=ones(size(examplesP,1),1);
    labelsS=ones(size(examplesS,1),1)+1;

    examples = [examplesP;examplesS];
    labels =[labelsP;labelsS];

    c1 = cvpartition(labelsP,'k',10);
    adb_acc = [];
    num_t_test = 10;
    for i=1:num_t_test
        tridx = c1.training(i);
        teidx = c1.test(i);
        extrain{1,i} = [examplesP(tridx,:);examplesS(tridx,:)];
        labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];
        extest{1,i} = [examplesP(teidx,:);examplesS(teidx,:)];
        labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

        %==============ADABOOST======================
        % % Use Adaboost to make a classifier
        % model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',50,'Tree');
        % % Make some test data
        % testdata = extest{1,i};

        % % Classify the testdata with the trained model
        % testclass=predict(model,testdata);
        % % Show result
        % corrects = sum(testclass == labelstest{1,i});
        % adb_acc(i) = corrects/length(labelstest{1,i});
        %==============End ADABOOST======================

        %=============NBAYESIAN=======================
        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        adb_acc(i) = 1- result{1,1};
        %=============End NBAYESIAN=======================

    end

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

% use AUC in mri_computeTvalues2: in file
% "transformIDM_selectActiveVoxels", line
% "[sortedPValues{c},sortedVoxels{c}] = sort(results{1,c}...
% - descend: 85%, [0.925 0.850 0.850 0.900 0.800 0.775]
% - ascend : 93%, [0.975 0.775 1.000 0.900 0.925 1.000]

% use Fisher in mri_computeTvalues2
% - descend: 100 features: 94.583%, [0.975 0.8 1 0.925 0.975 1]
%            200 features: 95.417%, [0.975 0.825 1 0.95 0.975 1]
% - ascend : 73.33% (WRONG)

% use abs(AUC) in mri_computeTvalues2
% - descend: 100 features: 94.583%, [0.975 0.850 1 0.900 0.95 1]
%            200 features: 95.833%, [1 0.825 1 0.975 0.95 1]
% - ascend : 70% (WRONG)