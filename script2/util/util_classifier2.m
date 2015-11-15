% this function use
%	extrain		labeltrain
% 	extest		labeltest
% 	classifier: "nbayes"	"adaM1"	"adaboost"
%	"adaboost" is as in book PR 

function acc = util_classifier2(extrain, extest, labeltrain, labeltest, classifier)
if strcmp(classifier,'nbayes')
	[classifier] = trainClassifier(extrain,labeltrain,'nbayes');
	[predictions] = applyClassifier(extest,classifier);
	[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labeltest);
	acc = 1- result{1,1};
elseif strcmp(classifier,'adaM1')
	model = fitensemble(extrain, labeltrain, 'AdaBoostM1',200,'Tree');
	testclass=predict(model,extest);
	corrects = sum(testclass == labeltest);
	acc = corrects/length(labeltest);
	disp(['Train adaM1 ']);
elseif strcmp(classifier,'adaboost')
	T_max=3000; % max number of base classifiers
	[pos_tot, thres_tot, sleft_tot, a_tot, P_tot,K] = boost_clas_coord(extrain', labeltrain', T_max);

	% 2.
	[y_out, P_error] = boost_clas_coord_out(pos_tot, thres_tot, sleft_tot, a_tot, P_tot,K,extest', labeltest');
	%figure(1), plot(P_error)
	corrects = sum(y_out == labeltest');
	acc = corrects/length(y_out);
	disp(['Train adaboost ']);

elseif strcmp(classifier,'knn') % default n = 3
    [classified]=k_nn_classifier(extrain',labelstrain',7,extest');
    [classif_error]=compute_error(labelstest',classified);
    acc = 1- classif_error;
end