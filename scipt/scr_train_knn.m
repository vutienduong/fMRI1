[i,d,m]=transformIDM_selectTrials(info,data,meta,find([info.cond]~=0)); % seletct non-noisey trials
[i,d,m]=transformIDM_selectActiveVoxels(i,d,m,20); % seletct 20 most active voxel
[examples,labels,expInfo] = idmToExamples_condLabel(i,d,m);  %create training data
%[classifier] = trainClassifier(examples,labels,'nbayes');   %train classifier
[classifier] = trainClassifier(examples,labels,'knn');   %train classifier
%[predictions] = applyClassifier(examples,classifier,'nbayes');       %test it
%[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labels);
%r = 1-result{1}  % rank accuracy
