t = cputime;
num_examples = size(Core,2);
examples = [];
half = 6;

labelsP = ones(half,1);
labelsS = ones(half,1) + 1;
labels = [labelsP;labelsS];
vector_dim = prod(size(Core{1,1}));

for i=1:num_examples
    coreTemp = Core{1,i};
    examples(i,:) = reshape(coreTemp, 1, vector_dim);
end

examplesP = examples(1:half,:);
examplesS = examples(half+1:half*2,:);
num_t_test = 10;
adb_acc = [];
c1 = cvpartition(labelsP,'k',num_t_test);

for i=1:num_t_test
    tridx = c1.training(i);
    teidx = c1.test(i);
    extrain{1,i} = [examplesP(tridx,:);examplesS(tridx,:)];
    labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];
    extest{1,i} = [examplesP(teidx,:);examplesS(teidx,:)];
    labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

%     model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',120,'Tree');
%     testclass=predict(model,extest{1,i});
%     % Show result
%     corrects = sum(testclass == labelstest{1,i});
%     adb_acc(i) = corrects/length(labelstest{1,i});
    
    [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
    [predictions] = applyClassifier(extest{1,i},classifier);
    [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
    adb_acc(1,i) = 1- result{1,1};
end

avg_acc = sum(adb_acc)/num_t_test;
e = cputime - t;
disp(['accuracy ', num2str(avg_acc) , ' | processing time ', num2str(e)]);