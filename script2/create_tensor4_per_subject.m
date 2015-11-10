function tensor = create_tensor4_per_subject()
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
    
    for i=1:length(dataP1)
        dataP(:,:,i) = dataP1{i,1}(1:54,:);
    end
    
    for i=1:length(dataS1)
        dataS(:,:,i) = dataS1{i,1}(1:54,:);
    end
    
    tensor{j,1} = dataP;
    tensor{num_subjects + j,1} = dataS;
end