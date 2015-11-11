% for each person , length(file_name)
% this file use FDR and nbayes with multi select the number of selected
% features uing FDR from E.g 50  to 100
clear; 
file_name = {'data-starplus-04799-v7', 'data-starplus-04847-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];

% all_nf_acc store accuracy of all nf(number of feature) setting
all_nf_acc = [];

% r1: minimum, r2: maximum, step: step
% Example: [50 100] step=5
r1 = 50; r2=100; step=10;

for nf=r1:step:r2
    disp(['No. feature : ', num2str(nf)]);
    for j=1:6
        clearvars -except j file_name all_acc nf all_nf_acc r1 r2 step;
        t = cputime;
        load(file_name{j});
        disp(['Complete load : ', file_name{j}]);

        trials=find([info.cond]>1); 
        [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
        % seperate P1st and S1st trials
        
        %num_per_ROI = 20;
        %ROIs = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
        %[info1,data1,meta1]=transformIDM_selectROIVoxels(info1,data1,meta1,ROIs);
        %[info1,data1,meta1] = nActivePerROICond(info,data,meta,ROIs,num_per_ROI);
        %[info1,data1,meta1] = createColToROI(info1,data1,meta1);
        %[info1,data1,meta1] = transformIDM_avgROIVoxels(info1,data1,meta1);
        
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

        examples=[examplesP;examplesS];
        labels=[labelsP;labelsS];

        [fdr,featrank]=sort(fdr,'descend');
        examplesPR = examplesP(:,featrank); 
        examplesSR = examplesS(:,featrank);
        examplesPS = examplesPR(:,1:nf); 
        examplesSS = examplesSR(:,1:nf);

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
    all_nf_acc((nf-r1)/step+1) = mean_all(1);
    %disp(['AVERAGE ACC', num2str(mean_all(1))]);
    %disp(['AVERAGE PROCESSING TIME ', num2str(mean_all(2))]);
end
plot(r1:step:r2, all_nf_acc);
% result
% choose ROIs: 4 secs, no ROIs: 8 secs
% 100 features: 96.25
% 200 features: 96.45