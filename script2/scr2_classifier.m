function acc = scr2_classifier(examples, labels, num_subjects, classify_type)
	c1 = cvpartition(labels,'k',num_subjects);
	switch(classify_type)
	case 'adaboost'
		for i=1:num_subjects
		    teidx = zeros(size(labels,1),1);
		    teidx(((i-1)*80 + 1):(i*80), :) = 1;
		    tridx = ones(size(labels,1),1) - teidx;

		    teidx = logical(teidx);
		    tridx = logical(tridx);

		    extrain{1,i} = examples(tridx,:);
		    labelstrain{1,i} = labels(tridx,:);
		    extest{1,i} = examples(teidx,:);
		    labelstest{1,i} = labels(teidx,:);

		    model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',120,'Tree');
		    testclass=predict(model, extest{1,i});
		    corrects = sum(testclass == labelstest{1,i});
		    acc(1,i) = corrects/length(labelstest{1,i});
		    disp(['Complete train ADABOOST', num2str(i)]);
		end
		acc = mean(acc);

	case 'gnb'
		for i=1:num_subjects
		    teidx = zeros(size(labels,1),1);
		    teidx(((i-1)*80 + 1):(i*80), :) = 1;
		    tridx = ones(size(labels,1),1) - teidx;

		    teidx = logical(teidx);
		    tridx = logical(tridx);

		    extrain{1,i} = examples(tridx,:);
		    labelstrain{1,i} = labels(tridx,:);
		    extest{1,i} = examples(teidx,:);
		    labelstest{1,i} = labels(teidx,:);

			[classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
			[predictions] = applyClassifier(extest{1,i},classifier);
			[result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
			acc(1,i) = 1- result{1};
		end
		acc = mean(acc);

	otherwise
		disp('this classifier is not supported. Please try again with adaboost or gnb')
	end
