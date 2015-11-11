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
    use_setting = 5;
    num_selected_feature = 240; % use for 1,2
    num_per_ROI = 50; % use for 3
    ROIs = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
    
    t = cputime;
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);
    
    
    if use_setting == 1 % ActiveVoxels
        % this cond is higher accuracy than (>0)
        % >0: 72% (240)
        % >1: 84% (240)
        % (>0 =>voxact => >1) 82%(240)
        [info1,data1,meta1] = activeVoxactCond(info,data,meta,ROIs,num_selected_feature);
        
    elseif use_setting == 2 % ActiveVoxact
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
    examplesP=[examplesP2;examplesP3];
    examplesS=[examplesS2;examplesS3];
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




