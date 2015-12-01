load('example4indexsPS2.mat');
num_subjects = 6;
labels = [ones(240,1); ones(240,1)+1];
all_acc = [];
all_std = [];

%1: FDR, 2: corre, 3: exhaust
use_setting = 2;

for numSelectedFeat=10:10:50
    for i=1:num_subjects
        use_ada = false;

        teidx = zeros(240,1);
        teidx((i*40-39):i*40) = 1;
        tridx = ones(240,1) - teidx;
        teidx = logical(teidx);
        tridx = logical(tridx);
        
        extrainP = examplesP(tridx,:);
        extrainS = examplesS(tridx,:);
        
        % FDR
        numfeat = size(extrainP,2);
        for ii=1:numfeat
            fdr(ii)= Fisher(extrainP(:,ii),extrainS(:,ii));
        end
        [fdr,featrank]=sort(fdr,'descend');
        
        if use_setting == 1
            selectedIndex = featrank(1:numSelectedFeat);
            examplesPR = extrainP(:,selectedIndex); 
            examplesSR = extrainS(:,selectedIndex);

            examplesPR2 = examplesP(teidx,selectedIndex);
            examplesSR2 = examplesS(teidx,selectedIndex);
        end
        

        % ttest
%         rho = 0.05;
%         [h] = ttest2(extrainP,extrainS,rho);
%         selectedIndex = find(h);

        % corre
        if use_setting > 1
            T =[fdr',featrank'];
            T = T(1:100,:);
            c1_train = extrainP';
            c2_train = extrainS';
            [p]= compositeFeaturesRanking (c1_train,c2_train,0.2,0.8,T);
            inds=sort(p(1:numSelectedFeat),'ascend');
            c1_train=c1_train(inds,:);
            c2_train=c2_train(inds,:);
            
            c1_test = examplesP(teidx,:)';
            c2_test = examplesS(teidx,:)';
            c1_test = c1_test(inds,:);
            c2_test = c2_test(inds,:);
            
            % exhausted
            if use_setting == 3
                [cLbest,Jmax]= exhaustiveSearch(c1_train,c2_train,'ScatterMatrices',[5]);
                c1_train = c1_train(cLbest,:); 
                c2_train = c2_train(cLbest,:);

                c1_test = c1_test(cLbest,:); 
                c2_test = c2_test(cLbest,:);
            end
    
            examplesPR = c1_train';
            examplesSR = c2_train';
            examplesPR2 = c1_test';
            examplesSR2 = c2_test';
            
        end
        
        extrain{1,i} = [examplesPR; examplesSR];
        extest{1,i} = [examplesPR2; examplesSR2];
        
        labelstrain{1,i} = [ones(200,1); ones(200,1)+1];
        labelstest{1,i} = [ones(40,1); ones(40,1)+1];

        % Bayes
        [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
        [predictions] = applyClassifier(extest{1,i},classifier);
        [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
        acc(1,i) = 1- result{1,1};

    end
    % disp(['Accuracy', num2str(mean(acc))]);
    % disp(['Std ', num2str(std(acc))]);
    all_acc = [all_acc; mean(acc)];
    all_std = [all_std; std(acc)];
    disp(['Accuracy', num2str(mean(acc))]);
end

% tot nhat la 0.8375 + 0.1294 tai N=14