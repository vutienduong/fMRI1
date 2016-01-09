% FDR 1: cal all data
% FDR 2: cal only training data

load('example4indexsPS2.mat');
num_subjects = 6;
labels = [ones(240,1); ones(240,1)+1];
all_acc = zeros(10,6);
all_acc2=[];
all_std = [];
%1: FDR, 2: corre, 3: exhaust
use_setting = 1;

for numSelectedFeat=10:10:100
    for i=1:num_subjects
        tridx = (i*40-39):i*40;
        
        dataP = examplesP(tridx,:);
        dataS = examplesS(tridx,:);

%         % FDR 1
%         numfeat = size(examplesP,2);
%         for ii=1:numfeat
%             fdr(ii)= Fisher(dataP(:,ii),dataS(:,ii));
%         end
%         [fdr,featrank]=sort(fdr,'descend');
% 
%         selectedIndex = featrank(1:numSelectedFeat);
%         examplesPR = dataP(:,selectedIndex); 
%         examplesSR = dataS(:,selectedIndex);
        
        num_t_test = 10;
        labelsP = labels(tridx);
        labelsS = labelsP + 1;
        c1 = cvpartition(labels(tridx),'k',10);
        for i2=1:num_t_test
            tridx = c1.training(i2);
            teidx = c1.test(i2);
            
            % FDR 2
%             extrainP = dataP(tridx,:);
%             extrainS = dataS(tridx,:);
%             numfeat = size(extrainP,2);
%             for ii=1:numfeat
%                 fdr(ii)= Fisher(extrainP(:,ii),extrainS(:,ii));
%             end
%             [fdr,featrank]=sort(fdr,'descend');
% 
%             selectedIndex = featrank(1:numSelectedFeat);
%             examplesPR = dataP(:,selectedIndex); 
%             examplesSR = dataS(:,selectedIndex);
            % end FDR 2
            examplesPR = dataP;
            examplesSR = dataS;
            
            extrain{1,i2} = [examplesPR(tridx,:);examplesSR(tridx,:)];
            labelstrain{1,i2} = [labelsP(tridx,:);labelsS(tridx,:)];
            extest{1,i2} = [examplesPR(teidx,:);examplesSR(teidx,:)];
            labelstest{1,i2} = [labelsP(teidx,:);labelsS(teidx,:)];

            [classifier] = trainClassifier(extrain{1,i2},labelstrain{1,i2},'nbayes');
            [predictions] = applyClassifier(extest{1,i2},classifier);
            [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i2});
            all_acc(i2,i) = 1- result{1,1};
        end


        % ttest
%         rho = 0.05;
%         [h] = ttest2(extrainP,extrainS,rho);
%         selectedIndex = find(h);

        % corre
%         if use_setting > 1
%             T =[fdr',featrank'];
%             T = T(1:100,:);
%             c1_train = extrainP';
%             c2_train = extrainS';
%             [p]= compositeFeaturesRanking (c1_train,c2_train,0.2,0.8,T);
%             inds=sort(p(1:numSelectedFeat),'ascend');
%             c1_train=c1_train(inds,:);
%             c2_train=c2_train(inds,:);
%             
%             c1_test = examplesP(teidx,:)';
%             c2_test = examplesS(teidx,:)';
%             c1_test = c1_test(inds,:);
%             c2_test = c2_test(inds,:);
%             
%             % exhausted
%             if use_setting == 3
%                 [cLbest,Jmax]= exhaustiveSearch(c1_train,c2_train,'ScatterMatrices',[5]);
%                 c1_train = c1_train(cLbest,:); 
%                 c2_train = c2_train(cLbest,:);
% 
%                 c1_test = c1_test(cLbest,:); 
%                 c2_test = c2_test(cLbest,:);
%             end
%     
%             examplesPR = c1_train';
%             examplesSR = c2_train';
%             examplesPR2 = c1_test';
%             examplesSR2 = c2_test';
%             
%         end
        
%         extrain{1,i} = [examplesPR; examplesSR];
%         extest{1,i} = [examplesPR2; examplesSR2];
%         
%         labelstrain{1,i} = [ones(200,1); ones(200,1)+1];
%         labelstest{1,i} = [ones(40,1); ones(40,1)+1];
% 
%         % Bayes
%         [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
%         [predictions] = applyClassifier(extest{1,i},classifier);
%         [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
%         acc(1,i) = 1- result{1,1};

    end
    % disp(['Accuracy', num2str(mean(acc))]);
    % disp(['Std ', num2str(std(acc))]);
    all_acc2 = [all_acc2; mean(all_acc)];
    all_std = [all_std; std(all_acc)];
    %disp(['Accuracy', num2str(mean(acc))]);
end

% FDR1: best 92.27
% FDR2: best 83.45
% neu k dung FDR: best 78.54
% NOTE: pp nay co the dung cho ca single and multi
% TODO: choose active voxel from generate example step, after calculate 4
% indexes