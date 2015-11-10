load data-starplus-05710-v7
[predictions] = applyClassifier(examples,classifier,'nbayes');       %test it
[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labels);
r = 1-result{1}  % rank accuracy