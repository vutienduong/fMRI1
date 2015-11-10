% this script use to test: 
% first, choose N most active voxels for each ROI, 
% then calculate Average of this N voxels, represent for a ROI
clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
examples = [];
labels = [];
num_subjects = 6;
N = 50;
use_avg_nFDR = true;

% create data (training + test)
for j=1:num_subjects
    clearvars -except j file_name examples labels num_subjects N use_avg_nFDR;
    load(file_name{j});

    % normalize (1) a voxel through all time points in a trial
    [info,data,meta] = transformIDM_normalizeTrials2(info,data,meta);

    trials=find([info.cond]>1);
    [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);

    % normalize (2) a
    [info1,data1,meta1] = transformIDM_normalizeImages(info1,data1,meta1); 

	% best ROIs
    ROIs = {'CALC' 'LDLPFC' 'LIPL' 'LIPS' 'LOPER'  'LT' 'LTRIA'};

    % select only voxels of 7 ROIs to decrease computational time when calculating FDR
    [info2,data2,meta2] = transformIDM_selectROIVoxels(info1,data1,meta1, ROIs);
    % [info2,data2,meta2] = transformIDM_avgROIVoxels(info1,data1,meta1,{'CALC' 'LDLPFC' 'LIPL' 'LIPS' 'LOPER'  'LT' 'LTRIA'});
	
	% create meta.roi and meta.colToROI
    [info2,data2,meta2] = createColToROI(info2,data2,meta2);

    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='S'));

    % seperate reading P vs S
    [infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
    [infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
    [infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
    [infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);

    % create a matrix for all voxel in all trial
    matrixP = [];
    matrixS = [];

    ntrials = length(infoP2);
    for j2=1:ntrials
        matrixP = [matrixP; dataP2{j2,1}; dataP3{j2,1}];
        matrixS = [matrixS; dataS2{j2,1}; dataS3{j2,1}];
    end

    % cal FDR
    nvoxels = metaP2.nvoxels;
    for i=1:nvoxels
        fdr(i)= Fisher(matrixP(:,i),matrixS(:,i));
    end

    matrix_sort = cell(1);
    % sort for each ROIs
    count = 1;
    i = 1;
    while count <= length(ROIs) && i < 26
		if strcmp(ROIs{1, count}, metaP2.rois(i).name)
			matrix_sort{count,1} = metaP2.rois(i).name;
			temp_fdr = fdr(metaP2.rois(i).columns);
			[temp_fdr,featrank]=sort(temp_fdr,'descend');
			temp_fdr = featrank(1:N);
			matrix_sort{count,2} = metaP2.rois(i).columns(temp_fdr);
			count = count + 1;
		end
		i = i + 1;
	end

	if use_avg_nFDR
		fdata = 1;
		finfo = 1;
		fmeta = 1;
		for i=1:length(matrix_sort)
			% TODO Note: in avgVoxelSubset, there is a normalization, care if it conclicts with norm(1) and norm(2)
			[tinfo,tdata,tmeta]=transformIDM_avgVoxelSubset(info1,data1,meta1,matrix_sort{i,2});
			if ( ~iscell(fdata) & ~isstruct(finfo) & ~isstruct(fmeta) )
				fdata = tdata;
				finfo = tinfo;
				fmeta = tmeta;
			else
				[finfo,fdata,fmeta] = mri_mergeIDM( finfo,fdata,fmeta,tinfo,tdata,tmeta );
			end
		end
		
		[infoP1,dataP1,metaP1]=transformIDM_selectTrials(finfo,fdata,fmeta,find([finfo.firstStimulus]=='P'));
		[infoS1,dataS1,metaS1]=transformIDM_selectTrials(finfo,fdata,fmeta,find([finfo.firstStimulus]=='S'));

		% seperate reading P vs S
		[infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
		[infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
		[infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
		[infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);
	else
		feature_index = [];
		for i=1:length(matrix_sort)
			feature_index = [feature_index matrix_sort{i,2}];
		end
		
		% update IDM as list of feature corresponding, here is 7*50 = 350 voxels
		[infoP2,dataP2,metaP2]= transformIDM_selectVoxelSubset(infoP2,dataP2,metaP2,feature_index);
		[infoP3,dataP3,metaP3]= transformIDM_selectVoxelSubset(infoP3,dataP3,metaP3,feature_index);
		[infoS2,dataS2,metaS2]= transformIDM_selectVoxelSubset(infoS2,dataS2,metaS2,feature_index);
		[infoS3,dataS3,metaS3]= transformIDM_selectVoxelSubset(infoS3,dataS3,metaS3,feature_index);
	end
	
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

    examples=[examples; examplesP; examplesS];
    labels=[labels; labelsP; labelsS];
    disp(['Complete create part ', num2str(j)]);
end

% classify
c1 = cvpartition(labels,'k',num_subjects);
for i=1:num_subjects
    % tridx = c1.training(i);
    % teidx = c1.test(i);
	use_ada = false;

    teidx = zeros(size(labels,1),1);
    teidx(((i-1)*80 + 1):(i*80), :) = 1;

    tridx = ones(size(labels,1),1) - teidx;

    teidx = logical(teidx);
    tridx = logical(tridx);

    extrain{1,i} = examples(tridx,:);
    labelstrain{1,i} = labels(tridx,:);
    extest{1,i} = examples(teidx,:);
    labelstest{1,i} = labels(teidx,:);

	if use_ada
        % Ada
        model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',120,'Tree');
        testclass=predict(model,extest{1,i});
        corrects = sum(testclass == labelstest{1,i});
        acc(1,i) = corrects/length(labelstest{1,i});
        disp(['Complete train ADABOOST', num2str(i)]);
    else
        % Bayes
        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        acc(1,i) = 1- result{1,1};
    end
end
disp(['Accuracy', num2str(mean(acc))]);
disp(['Std ', num2str(std(acc))]);



% ======================= RESULTS ===============================
% N=50, Bayes, use 50 voxels (use FDR to select) for each ROI, total 350 voxels 
% 	=> 87.5% +- 14.81 (1; 0.5875; 0.975; 0.9; 0.8875; 0.9)
% N=50, Bayes, use 1 supervoxel for each ROI (average of 50 voxels selected by FDR), total 7 voxels 
% 	=> 57.71% +- 7.26