clear;
load ionosphere;
rng(2); % For reproducibility
ClassTreeEns = fitensemble(X,Y,'AdaBoostM1',100,'Tree',...
    'Holdout',0.5);
genError = kfoldLoss(ClassTreeEns,'Mode','Cumulative');
plot(genError);
xlabel('Number of Learning Cycles');
ylabel('Generalization Error');
