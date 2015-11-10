function tensor = create_tensor2_based_active()
clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7', 'data-starplus-04820-v7'};
% ,...
% 'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
num_subjects = length(file_name);

min_voxel = 10000;
ROIs = {'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
for j=1:num_subjects
	load(file_name{j});
    [info,data,meta] = transformIDM_selectROIVoxels(info,data,meta,ROIs);
	if size(data{1,1}, 2) < min_voxel
		min_voxel = size(data{1,1}, 2);
	end
end

n20 = 20;
tensorP2 = cell(n20,1);
tensorP3 = cell(n20,1);
tensorS2 = cell(n20,1);
tensorS3 = cell(n20,1);

for j=1:num_subjects
    load(file_name{j});

    [info1,data1,meta1] = transformIDM_selectROIVoxels(info,data,meta,ROIs);
    [info1,data1,meta1] = transformIDM_selectActiveVoxact(info1,data1,meta1,min_voxel);
    trials=find([info.cond]>1);
    [info1,data1,meta1] = transformIDM_selectTrials(info1,data1,meta1,trials);
    [info1,data1,meta1] = transformIDM_normalizeTrials2(info1,data1,meta1);
    
    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));

    % seperate reading P vs S
    [infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
    [infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
    [infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
    [infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);
    
    for ntrial=1:metaP2.ntrials
    	tensorP2{ntrial,1}(j, :, :) = dataP2{ntrial,1};
        tensorP3{ntrial,1}(j, :, :) = dataP3{ntrial,1};
        tensorS2{ntrial,1}(j, :, :) = dataS2{ntrial,1};
        tensorS3{ntrial,1}(j, :, :) = dataS3{ntrial,1};
    end
end

tensor = [tensorP2; tensorP3; tensorS2; tensorS3];