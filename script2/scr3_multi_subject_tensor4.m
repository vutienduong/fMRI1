t = cputime;
half_examples = size(Core,2)/2;
examples = [];

labelsP = ones(half_examples,1) + 1;
labelsS = ones(half_examples,1);
labels = [labelsP;labelsS];
vector_dim = prod(size(Core{1,1}));

for i=1:half_examples*2
    coreTemp = Core{1,i};
    examples(i,:) = reshape(coreTemp, 1, vector_dim);
end

examplesP = examples(1:half_examples,:);
examplesS = examples(half_examples + 1 : half_examples * 2, :);

numfeat = size(examplesP,2);
for i=1:numfeat
    fdr(i)= Fisher(examplesP(:,i),examplesS(:,i));
end

[fdr,featrank]=sort(fdr,'descend');
examplesPR = examplesP(:,featrank); 
examplesSR = examplesS(:,featrank);

avg_acc = [];
select_num = 100;
for numOfFeature = 1:select_num
    examplesPS = examplesPR(:,1:numOfFeature); 
    examplesSS = examplesSR(:,1:numOfFeature);
    examples = [examplesPS; examplesSS];

    adb_acc = [];

    for i=1:half_examples*2
        tridx = ones(half_examples*2,1);
        tridx(i,1) = 0;
        tridx = logical(tridx);
        teidx = not(tridx);

        extrain{1,i} = examples(tridx,:);
        labelstrain{1,i} = labels(tridx,:);
        extest{1,i} = examples(teidx,:);
        labelstest{1,i} = labels(teidx,:);

        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        adb_acc(1,i) = 1- result{1,1};

    %     model = fitensemble(extrain{1,i}, labelstrain{1,i}, 'AdaBoostM1',120,'Tree');
    %     testclass=predict(model,extest{1,i});
    %     corrects = sum(testclass == labelstest{1,i});
    %     adb_acc(i) = corrects/length(labelstest{1,i});
    end

    avg_acc(numOfFeature) = sum(adb_acc)/(half_examples*2);
end
plot(1:select_num, avg_acc);
e = cputime - t;
disp(['accuracy ', num2str(avg_acc(9)),num2str(avg_acc(3)),num2str(avg_acc(4)) , ' | processing time ', num2str(e)]);