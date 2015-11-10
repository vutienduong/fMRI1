clear;
% >> TEST Tensor
% X = create_tensor;
% [Core, FACT] = decompose_tensor_HONMF(X, 3);


% >> TEST mri_mergeIDM
% load data-starplus-04799-v7.mat
% info1 = info;
% data1 = data;
% meta1 = meta;

% load data-starplus-04820-v7.mat
% info2 = info;
% data2 = data;
% meta2 = meta;

% [info3,data3,meta3] = mri_mergeIDM( info1,data1,meta1,info2,data2,meta2 );


% >> TEST create_tensor2_based_active
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7', 'data-starplus-04820-v7'};
num_subjects = length(file_name);

for j=1:1
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

    examples=[examplesP;examplesS];
    labels=[labelsP;labelsS];
end