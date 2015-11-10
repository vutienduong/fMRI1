function acc = util_classifier(tridx, teidx, examples, labels)
% create Xtrain, Ytrain, Xtest, Ytest (Y is label)
extrain = examples(tridx,:);
labelstrain = labels(tridx,:);
extest = examples(teidx,:);
labelstest = labels(teidx,:);

% classify using Bayes
[classifier] = trainClassifier(extrain,labelstrain,'nbayes');
[predictions] = applyClassifier(extest,classifier);
[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest);
% accuracy
acc = 1- result{1,1};