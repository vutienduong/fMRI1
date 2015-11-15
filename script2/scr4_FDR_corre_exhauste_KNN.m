% for each person , length(file_name)
% FDR correlation exhauste
clear; 
file_name = {'data-starplus-04799-v7', 'data-starplus-04847-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_acc = [];
for j=1:6
    clearvars -except j file_name all_acc;
    t = cputime;
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);

    trials=find([info.cond]>1); 
    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
    % seperate P1st and S1st trials
    [info1,data1,meta1]=transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));
    %trials=find([info1.cond]>0); 
    %[info1,data1,meta1]=transformIDM_selectTrials(info1,data1,meta1,trials);
 
    % seperate reading P vs S
    [infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
    [infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
    [infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
    [infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);

    % convert to examples
    [examplesP2,labelsP2,exInfoP2]=idmToExamples_condLabel(infoP2,dataP2,metaP2);
    [examplesP3,labelsP3,exInfoP3]=idmToExamples_condLabel(infoP3,dataP3,metaP3);
    [examplesS2,labelsS2,exInfoS2]=idmToExamples_condLabel(infoS2,dataS2,metaS2);
    [examplesS3,labelsS3,exInfoS3]=idmToExamples_condLabel(infoS3,dataS3,metaS3);

    % combine examples and create labels.  Label 'picture' 1, label 'sentence' 2.
    examplesP=[examplesP2;examplesP3];
    examplesS=[examplesS2;examplesS3];
    labelsP=ones(size(examplesP,1),1);
    labelsS=ones(size(examplesS,1),1)+1;

    c1 = cvpartition(labelsP,'k',10);
    adb_acc = [];
    num_t_test = 10;
    for i=1:num_t_test
        tridx = c1.training(i);
        teidx = c1.test(i);

        examplesP_train = examplesP(tridx,:);
        examplesS_train = examplesS(tridx,:);
        
        % Fisher ===========================
        numfeat = size(examplesP_train,2);
        for ii=1:numfeat
            fdr(ii)= Fisher(examplesP_train(:,ii),examplesS_train(:,ii));
        end
        [fdr,featrank]=sort(fdr,'descend');
        % end Fisher ===========================

        % Transport
        T =[fdr',featrank'];
        T = T(1:50,:); % TEST
        c1_train = examplesP_train';
        c2_train = examplesS_train';
        disp(['DONE !', num2str(cputime-t)]); %  Transport

        % 2. The scalar feature ranking technique which employs FDR
        % in conjunction with a feature correlation measure.
        [p]= compositeFeaturesRanking (c1_train,c2_train,0.2,0.8,T);
        % TEST
        disp(['DONE !', num2str(cputime-t)]); % compositeFeaturesRanking 

        % 3. In order to reduce the dimensionality of the feature space, work with the 7 highest ranked features
        inds=sort(p(1:10),'ascend');
        c1_train= c1_train(inds,:);
        c2_train= c2_train(inds,:);

        % 4. Choose the best feature combination consisting of three features (out of the previously selected seven), using the exhaustive search method.
        [cLbest,Jmax]= exhaustiveSearch(c1_train,c2_train,'ScatterMatrices',[5]);
%         fprintf('\n Exhaustive Search -> Best of ten: ');
%         fprintf('(%d) ',inds(cLbest));
%         
%         [cLbest,Jmax]=SequentialForwardSelection(c1_train,c2_train,'ScatterMatrices',5);
%         fprintf('\n Sequential Forward Selection -> Best of ten: ');
%         fprintf('(%d) ',inds(cLbest));
% 
%         [cLbestSBS,JSBS]=SequentialBackwardSelection(c1_train(inds,:),c2_train(inds,:),'ScatterMatrices',10);
%         fprintf('\n Sequential Backward Selection -> Best of ten: ');
%         fprintf('(%d) ',inds(cLbestSBS));
% 
%         [cLbestSFFS,JSFFS]=SequentialForwardFloatingSelection(c1_train(inds,:),c2_train(inds,:),'ScatterMatrices',10);
%         fprintf('\n Floating Search Method -> Best of ten: ');
%         fprintf('(%d) ',inds(cLbestSFFS));
%         fprintf('\n');
        
        
        
        % TEST
        disp(['DONE !', num2str(cputime-t)]); %  exhaustiveSearch
        
        

        % 5. Form the resulting training dataset (using the best feature combination), along with the corresponding class labels.
        c1_train = c1_train(cLbest,:); 
        c2_train = c2_train(cLbest,:);
        
       
        % create train set with cLbest features
        extrain{1,i} = [c1_train'; c2_train'];
        labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];
        
         %test follow
        c1_test = examplesP(teidx,:)';
        c2_test = examplesS(teidx,:)';
        c1_test = c1_test(inds,:);
        c2_test = c2_test(inds,:);
        c1_test = c1_test(cLbest,:); 
        c2_test = c2_test(cLbest,:);

        % create test set with cLbest features
        extest{1,i} = [c1_test'; c2_test'];
        labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

%         [classifier] = trainClassifier(extrain{1,i},labelstrain{1,i},'nbayes');
%         [predictions] = applyClassifier(extest{1,i},classifier);
%         [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',labelstest{1,i});
%         adb_acc(1,i) = 1- result{1,1};

        [classified]=k_nn_classifier(extrain{1,i}',labelstrain{1,i}',9,extest{1,i}');
        [classif_error]=compute_error(labelstest{1,i}',classified);
        adb_acc(1,i) = 1- classif_error;
        disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);

        % (c) Apply the Leave One Out method on the k-NN classifier (k = 3) and compute the error.
%         [M,N]=size(AllDataset);
%         for i=1:N
%             dec(i)=k_nn_classifier([AllDataset(:,1:i-1) AllDataset(:,i+1:N)], [Labels(1,1:i-1) Labels(1,i+1:N)],3,AllDataset(:,i));
%         end
%         LOO_error=sum((dec~=Labels))/N;
    end

    avg_acc = sum(adb_acc)/num_t_test;
    e = cputime - t;
    t = cputime;
    all_acc = [all_acc; avg_acc e];
    disp(['accuracy ', num2str(avg_acc) , ' | processing time ', num2str(e)]);

end
mean_all = mean(all_acc);
disp(['AVERAGE ACC', num2str(mean_all(1))]);
disp(['AVERAGE PROCESSING TIME ', num2str(mean_all(2))]);

% select ROIs, 100 FDR, 15 correlation, 10 comb:  80.417% 45.26s [0.7125;0.98750;0.9125;0.55;0.75;0.9125]
% select ROIs, 100 FDR, 15 correlation:  84.375% 38.23s [0.8125;0.98750;0.8750;0.65;0.8625;0.8750]
% select ROIs, 100 FDR, 30 correlation:  86.458% 38.32s [0.8375;0.98750;0.9625;0.625;0.9;0.8750]
% select ROIs, 50 FDR, 20 correlation:  87.292% 32.3s
% select ROIs, 100 FDR, 50 correlation:  89.96% 38.72s [0.8750;1;0.9625;0.7;0.925;0.8750]
% no ROIs, 100 FDR, 50 correlation:  86.45% 84.70s [0.8875;1;0.9375;0.6125;0.8875;0.8625]

% Exhaustive Search -> Best of ten:             (4831) (7311) (18819) (20706) (22523) (22564) (22714) (24589) (26310) (26312) 
% Sequential Forward Selection -> Best of ten:  (4831) (7311) (18819) (20704) (22564) (22714) (24589) (26310) (26312) (26315) 
% Sequential Backward Selection -> Best of ten: (4831) (7311) (18819) (20706) (22523) (22564) (22714) (24589) (26310) (26312) 
% Floating Search Method -> Best of ten:        (4831) (7311) (18819) (20706) (22523) (22564) (22714) (24589) (26310) (26312) 

% 50 10 5. knn(3) 77.5%
% 50 10 5. knn(7) 83.125% 33s
% 50 10 5. knn(9) 81.458% 33s
% 50 10 5, SFS knn(9) 78.33% 33s
% 50 10 5. nbayes 81.25% 33.9s


    