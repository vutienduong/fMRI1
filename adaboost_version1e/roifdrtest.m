clear; 
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];
for j=1:6
	clearvars -except j file_name all_acc;
	t = cputime;
	load(file_name{j});
	disp(['Complete load : ', file_name{j}]);
	trials=find([info.cond]>1);
	[info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
	[info2,data2,meta2]=transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
	[infoP1,dataP1,metaP1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='P'));
	[infoS1,dataS1,metaS1]=transformIDM_selectTrials(info2,data2,meta2,find([info2.firstStimulus]=='S'));

	% [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info2,data2,meta2,find([info1.cond]==2));
 %    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info2,data2,meta2,find([info1.cond]==3));
	[infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
	[infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
	[infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
	[infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);
	[examplesP2,labelsP2,exInfoP2]=idmToExamples_condLabel(infoP2,dataP2,metaP2);
	[examplesP3,labelsP3,exInfoP3]=idmToExamples_condLabel(infoP3,dataP3,metaP3);
	[examplesS2,labelsS2,exInfoS2]=idmToExamples_condLabel(infoS2,dataS2,metaS2);
	[examplesS3,labelsS3,exInfoS3]=idmToExamples_condLabel(infoS3,dataS3,metaS3);
	examplesP=[examplesP2;examplesP3];
	examplesS=[examplesS2;examplesS3];
	examples=[examplesP;examplesS];

	labelsP = ones(size(examplesP, 1),1);
	labelsS = ones(size(examplesS, 1),1)+1;

	numfeat = size(examplesP,2);
	for i=1:numfeat
		fdr(i)= Fisher(examplesP(:,i),examplesS(:,i));
	end
	[fdr,featrank]=sort(fdr,'descend');
	examplesPR = examplesP(:,featrank);
	examplesSR = examplesS(:,featrank);
	examplesPS = examplesPR(:,1:50);
	examplesSS = examplesSR(:,1:50);

	% examplesPS = examplesP;
 %    examplesSS = examplesS;

	c1 = cvpartition(labelsP,'k',10);
	for i=1:10
		tridx = c1.training(i);
		teidx = c1.test(i);
		extrain{1,i} = [examplesPS(tridx,:);examplesSS(tridx,:)];
		labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];
		extest{1,i} = [examplesPS(teidx,:);examplesSS(teidx,:)];
		labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

		[classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
		[predictions] = applyClassifier(extest{1,i},classifier);
		[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
		adb_acc(i) = 1- result{1,1};
	end

	avg_acc = sum(adb_acc)/10;
	e = cputime - t;
	t = cputime;
	all_acc = [all_acc; avg_acc e];
	disp(['accuracy ', num2str(avg_acc) , ' | processing time ', num2str(e)]);
end