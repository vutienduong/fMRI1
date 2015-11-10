function tensor = create_tensor5_per_subject_total24()
clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7', 'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
num_subjects = length(file_name);
n12 = num_subjects*2;
tensor = cell(n12,1); 

for j=1:num_subjects
    clearvars -except j file_name num_subjects tensor;
    load(file_name{j});
    ROIs = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
    [info,data,meta] = transformIDM_selectROIVoxels( info , data, meta, ROIs);
    trials=find([info.cond]>1);
    [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);
    [info1,data1,meta1] = transformIDM_normalizeTrials2(info1,data1,meta1);
    
    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));
    
    [infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
    [infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
    [infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
    [infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);
    
    for i=1:length(dataP2)
        tempP2(:,:,i) = dataP2{i,1};
    end
    
    for i=1:length(dataS2)
        tempS2(:,:,i) = dataS2{i,1};
    end
    
    for i=1:length(dataP3)
        tempP3(:,:,i) = dataP3{i,1};
    end
    
    for i=1:length(dataS3)
        tempS3(:,:,i) = dataS3{i,1};
    end
    
    
    tensor{j,1} = tempP2;
    tensor{num_subjects + j,1} = tempP3;
    tensor{num_subjects*2 + j,1} = tempS2;
    tensor{num_subjects*3 + j,1} = tempS3;
    
    % tensor 16 x 1715 x 20 (16 time point, 1715 voxel, 20 trial)
end