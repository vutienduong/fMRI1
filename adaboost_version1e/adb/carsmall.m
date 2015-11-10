clear
load carsmall
X = [Cylinders,Displacement,Horsepower,Weight];
xnames = {'Cylinders','Displacement','Horsepower','Weight'};
RegTreeTemp = templateTree('Surrogate','On');
RegTreeEns = fitensemble(X,MPG,'LSBoost',100,RegTreeTemp,...
    'PredictorNames',xnames);
predMPG = predict(RegTreeEns,[4 200 150 3000])