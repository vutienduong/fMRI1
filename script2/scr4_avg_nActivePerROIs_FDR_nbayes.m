% for each person , length(file_name)
% this file use FDR and nbayes with select nActiveVoxels/ROIs, then
% average (7), then use FDR (less than 7)
clear; 
file_name = {'data-starplus-04799-v7', 'data-starplus-04847-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];
for j=1:6
    clearvars -except j file_name all_acc nf all_nf_acc r1 r2 step;
    t = cputime;
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);

    num_per_ROI = 20;
    ROIs = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
    
    [examplesP, examplesS] = avg_nActivePerROICondFixed(info,data,meta, ROIs, num_per_ROI);
    labelsP=ones(size(examplesP,1),1);
    labelsS=ones(size(examplesS,1),1)+1;

    % Fisher ===========================
    numfeat = size(examplesP,2);

    for i=1:numfeat
        fdr(i)= Fisher(examplesP(:,i),examplesS(:,i));
    end

    examples=[examplesP;examplesS];
    labels=[labelsP;labelsS];

    nf = 5;
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
disp(['AVERAGE ACC', num2str(mean_all(1))]);
disp(['AVERAGE PROCESSING TIME ', num2str(mean_all(2))]);
    
% result
% 63%(2) - 70%(5)