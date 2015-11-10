clear;
file_name = {'data-starplus-04847-v7', 'data-starplus-04820-v7', 'data-starplus-04847-v7',...
    'data-starplus-05675-v7', 'data-starplus-05680-v7', 'data-starplus-05710-v7'};
load(file_name{1});
nToKeep = 20;

%CALC
% [info1,data1,meta1] = transformIDM_selectROIVoxels(info,data,meta,{'CALC'});
 trials=find([info.cond]~=0); 
    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
[info2,data2,meta2] = transformIDM_selectActiveVoxact(info1,data1,meta1,nToKeep, [2,3]);

% IDM_information=IDMinformation( info1,data1,meta1,meta1.study );
% % nvoxels=IDM_information.nVoxels;
% % if nToKeep < nvoxels
% %     [info1,data1,meta1] = transformIDM_selectActiveVoxact(info1,data1,meta1,nToKeep);
% % end

% [examples1,labels1,expInfo1] = idmToExamples_condLabel(info1,data1,meta1);

% %LIT
% [info2,data2,meta2] = transformIDM_selectROIVoxels(info,data,meta,{'LIT'});

% IDM_information=IDMinformation( info2,data2,meta2,meta2.study );
% % nvoxels=IDM_information.nVoxels;
% % if nToKeep < nvoxels
% %     [info2,data2,meta2] = transformIDM_selectActiveVoxact(info2,data2,meta2,nToKeep);
% % end

% %[info2,data2,meta2] = transformIDM_avgROIVoxels(info2,data2,meta2,{'LIT'});
% [examples2,labels2,expInfo2] = idmToExamples_condLabel(info2,data2,meta2);

% %[rinfo,rdata,rmeta]=transformIDM_mergeMulti(info1,data1,meta1,info2,data2,meta2,info3,data3,meta3);
% [examples, labels, expInfo] = mergeExamples(examples1, labels1, expInfo1, examples2, labels2, expInfo2);
% labelCond = labels > 0;
% labels = labels(labelCond);
% examples = examples(labelCond, :);
% [classifier] = trainClassifier(examples,labels,'nbayes');   %train classifier

% [predictions] = applyClassifier(examples,classifier,'nbayes');
% [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labels);
% 1-result{1}
