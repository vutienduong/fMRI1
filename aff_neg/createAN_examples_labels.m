function [examplesP1, examplesS1] = createAN_examples_labels(examplesP, examplesS)
% load label AFF (1), NEG (-1) from "aff_neg_data2"
[labelsPS2, labelsPS3] = aff_neg_data2();
labels   = [labelsPS2'; labelsPS3'; labelsPS2'; labelsPS3'];
examples = [examplesP; examplesS];
affidx = labels==1;

% examplesP now is affirmative, examplesS now is negative
examplesP1 = examples(affidx,:);
examplesS1 = examples(not(affidx),:);