load ionosphere
ClassTreeEns = fitensemble(X,Y,'AdaBoostM1',100,'Tree');
rsLoss = resubLoss(ClassTreeEns,'Mode','Cumulative');
plot(rsLoss);
xlabel('Number of Learning Cycles');
ylabel('Resubstitution Loss');