function tensor = create_tensor()
clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7', 'data-starplus-04820-v7'};
% ,...
% 'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
examples = [];
labels = [];
num_subjects = length(file_name);
max_num_voxel = 0;

for j=1:num_subjects
	load(file_name{j});
	if size(data{1,1}, 2) > max_num_voxel
		max_num_voxel = size(data{1,1}, 2);
	end
end
disp(['max_num is ', num2str(max_num_voxel)]);

num_trials_each_stimuli = 20;
tensorP2 = cell(num_trials_each_stimuli);
tensorP3 = cell(num_trials_each_stimuli);
tensorS2 = cell(num_trials_each_stimuli);
tensorS3 = cell(num_trials_each_stimuli);

% create tensor, fill by 0
% train, test by k-folding
for j=1:num_subjects
    % clearvars -except j file_name num_subjects;
    load(file_name{j});

    trials=find([info.cond]>1);
    [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);

    [info2,data2,meta2] = transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});

    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='S'));

    % seperate reading P vs S
    [infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
    [infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
    [infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
    [infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);


    for ntrial=1:metaP2.ntrials
    	add_temp = dataP2{ntrial,1};
        add_temp = abs(add_temp); % test
    	if size(add_temp, 2) < max_num_voxel
    		add_temp = [add_temp zeros(size(add_temp, 1), max_num_voxel - size(add_temp, 2))];
    	end
    	tensorP2{ntrial}(j, :, :) = add_temp;
    end

    for ntrial=1:metaP3.ntrials
    	add_temp = dataP2{ntrial,1};
        add_temp = abs(add_temp); % test
    	if size(add_temp, 2) < max_num_voxel
    		add_temp = [add_temp zeros(size(add_temp, 1), max_num_voxel - size(add_temp, 2))];
    	end
    	tensorP3{ntrial}(j, :, :) = add_temp;
    end

    for ntrial=1:metaS2.ntrials
    	add_temp = dataP2{ntrial,1};
        add_temp = abs(add_temp); % test
    	if size(add_temp, 2) < max_num_voxel
    		add_temp = [add_temp zeros(size(add_temp, 1), max_num_voxel - size(add_temp, 2))];
    	end
    	tensorS2{ntrial}(j, :, :) = add_temp;
    end

    for ntrial=1:metaS3.ntrials
    	add_temp = dataP2{ntrial,1};
        add_temp = abs(add_temp); % test
    	if size(add_temp, 2) < max_num_voxel
    		add_temp = [add_temp zeros(size(add_temp, 1), max_num_voxel - size(add_temp, 2))];
    	end
    	tensorS3{ntrial}(j, :, :) = add_temp;
    end
end

tensor = [tensorP2; tensorP3; tensorS2; tensorS3];