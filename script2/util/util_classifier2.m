% this function use
%	extrain		labeltrain
% 	extest		labeltest
% 	classifier: "nbayes"	"adaM1"	"adaboost"
%	"adaboost" is as in book PR 

function acc = util_classifier2(extrain, extest, labeltrain, labeltest, classifier, varargin)
extrain_T = extrain';
labeltrain_T = labeltrain';
extest_T = extest';
labeltest_T = labeltest';

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
	T_max=1000; % max number of base classifiers
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
    
elseif strcmp(classifier,'svm') || strcmp(classifier,'perce')

    C=0.5;
    tol=0.001;
    steps=100000;
    eps=10^(-10);
    method=1;
    max_iter = 30000;
    use_kernel = 'poly'; %linear, rbf, poly
    
    if strcmp(use_kernel,'linear')
        kernel='linear';
        kpar1=0;
        kpar2=0;
        if strcmp(classifier,'svm')
            [alpha, w0, w, evals, stp, glob] = SMO2(extrain, labeltrain,kernel, kpar1, kpar2, C, tol, steps, eps, method);

            % Compute the classification error on the test set
            Pe_te=sum((2*(w*extest_T-w0>0)-1).*labeltest_T<0)/length(labeltest_T);
            acc = 1- Pe_te;
        end
    elseif  strcmp(use_kernel,'rbf')
        kernel='rbf';
        kpar1=1.5;
        kpar2=0;
    elseif strcmp(use_kernel,'poly')
        kernel='poly';
        kpar1=1;
        kpar2=3;
    end

    % Classification of the test set
    % Computing the test error
    if strcmp(use_kernel,'rbf') || strcmp(use_kernel,'poly') ...
        || (strcmp(use_kernel,'linear') && strcmp(classifier,'perce'))
        if strcmp(classifier,'svm')
            [alpha, b, w, evals, stp, glob] = SMO2(extrain, labeltrain, kernel, kpar1, kpar2, C, tol, steps, eps, method);
            
            X_sup = extrain_T(:,alpha'~=0);
            alpha_sup=alpha(alpha~=0)';
            y_sup=labeltrain_T(alpha~=0);

            % Classification of the test set
            for i=1:size(extest,1)
                t=sum((alpha_sup.*y_sup).*CalcKernel(X_sup',extest_T(:,i)',kernel,kpar1,kpar2)')-b;
                if(t>0)
                    out_test(i)=1;
                else
                    out_test(i)=-1;
                end
            end

            % Computing the test error
            Pe_te=sum(out_test.*labeltest_T<0)/length(labeltest_T);
            acc = 1- Pe_te;
        elseif strcmp(classifier,'perce')
            [a,iter,count_misclas]=kernel_perce(extrain_T,labeltrain_T,kernel,kpar1,kpar2,max_iter);
            % Compute the test error
            for i=1:size(extest,1)
                K=CalcKernel(extrain,extest_T(:,i)',kernel,kpar1,kpar2)';
                out_test(i)=sum((a.*labeltrain_T).*K)+sum(a.*labeltrain_T);
            end
            Pe_te=sum(out_test.*labeltest_T<0)/length(labeltest_T);
            acc = 1- Pe_te;
        end
        
    end
elseif strcmp(classifier,'nn')
    iter=9000; %Number of iterations
    code=1; %Code for the chosen training algorithm
    k=2; %number of hidden layer nodes
    lr=.01; %learning rate
    % par_vec=[lr 0 1.05 0.7 1.04];
    par_vec=[lr 0 0 0 0];
    [net,tr]=NN_training(extrain_T,labeltrain_T,k,code,iter,par_vec);

    % Compute the training and the test errors
    Pe_te=NN_evaluation(net,extest_T,labeltest_T)
    acc = 1- Pe_te;
end