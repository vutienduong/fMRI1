% P3 va S2: SP
% P2 va S3: PS
clear; 
file_name = {'data-starplus-04799-v7', 'data-starplus-04847-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
all_settings_acc = [];
all_settings_acc_details = [];

for nf_j = [30 50 100]
    clearvars -except j file_name all_settings_acc all_settings_acc_details nf_j;
    all_acc = [];
    for j=1:6% 2:6 %2:2:4
        clearvars -except j file_name all_acc all_settings_acc all_settings_acc_details nf_j;
        t = cputime;
        load(file_name{j});
        disp(['Complete load : ', file_name{j}]);
        
        % FALSE: P3 va S2: S2P3
        % TRUE:  P2 va S3: P2S3
        test_PS = false;

        % collect the non-noise and non-fixation trials
        trials=find([info.cond]>1); 
        [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);

        % seperate P1st and S1st trials
        [info1,data1,meta1]=transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
        
        [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
        [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));

        % seperate reading P vs S
        if test_PS %P2S3
            [infoP2,dataP2,metaP2]= transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
            [infoS2,dataS2,metaS2]= transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);
        else %S2P3
            [infoP2,dataP2,metaP2]= transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
            [infoS2,dataS2,metaS2]= transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);
        end

        % convert to examples
        [examplesP2,labelsP2,exInfoP2]=idmToExamples_condLabel(infoP2,dataP2,metaP2);
        [examplesS2,labelsS2,exInfoS2]=idmToExamples_condLabel(infoS2,dataS2,metaS2);
        %[examplesP3,labelsP3,exInfoP3]=idmToExamples_condLabel(infoP3,dataP3,metaP3);
        %[examplesS3,labelsS3,exInfoS3]=idmToExamples_condLabel(infoS3,dataS3,metaS3);

        % combine examples and create labels.  Label 'picture' 1, label 'sentence' 2.
        % examplesP=[examplesP2;examplesP3];
        % examplesS=[examplesS2;examplesS3];
        examplesP = examplesP2;
        examplesS = examplesS2;
        labelsP=ones(size(examplesP,1),1);
        labelsS=ones(size(examplesS,1),1)+1;

        examples=[examplesP;examplesS];
        labels=[labelsP;labelsS];



        c1 = cvpartition(labelsP,'k',10);
        adb_acc = [];
        num_t_test = 10;

        % use Bayes or Ada here
        % 1)nbayes(1,2)   2)adaM1(1,2)     3)adaboost(1,-1) 4) knn(1,2) 
        % 5)svm (1,-1) 6)kernel perceptron(1,-1) 7) nn(1,-1): neuronnetwork 

        classifier = 'nbayes';

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

            % choosing 100 features based on FDR values
            nf = nf_j;
            examplesPR = examplesP(:,featrank); 
            examplesSR = examplesS(:,featrank);
            examplesPS = examplesPR(:,1:nf); 
            examplesSS = examplesSR(:,1:nf);

            % create train set with 100 features based on examplesPS & examplesSS
            extrain{1,i} = [examplesPS(tridx,:); examplesSS(tridx,:)];
            labelstrain{1,i} = [labelsP(tridx,:);labelsS(tridx,:)];

            % create test set with 100 features based on examplesPS & examplesSS
            extest{1,i} = [examplesPS(teidx,:); examplesSS(teidx,:)];
            labelstest{1,i} = [labelsP(teidx,:);labelsS(teidx,:)];

            % classify
            if strcmp(classifier,'adaM1')
                adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'adaM1');
            elseif strcmp(classifier,'nbayes')
                adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'nbayes');
            elseif strcmp(classifier,'adaboost')
                adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'adaboost');
                % disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
            elseif strcmp(classifier,'knn')
                adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'knn');
                disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
            elseif strcmp(classifier,'svm')
                adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'svm');
                disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
            elseif strcmp(classifier,'perce')
                adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'perce');
                disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
            elseif strcmp(classifier,'nn')
                adb_acc(i) = util_classifier2(extrain{1,i}, extest{1,i}, labelstrain{1,i}, labelstest{1,i}, 'nn');
                disp(['TEMP accuracy ', num2str(adb_acc(1,i))]);
            end
        end
        avg_acc = sum(adb_acc)/num_t_test;
        e = cputime - t;
        t = cputime;
        all_acc = [all_acc; avg_acc e];
        disp(['accuracy ', num2str(avg_acc) , ' | processing time ', num2str(e)]);
    end
    mean_all = mean(all_acc);
    disp(['AVERAGE ACC ', num2str(mean_all(1))]);
    disp(['AVERAGE PROCESSING TIME ', num2str(mean_all(2))]);
    
    % store over each setting
    all_settings_acc = [all_settings_acc; mean_all];
    all_settings_acc_details = [all_settings_acc; all_acc];
end


% P2S3, ROIs 30, nbayes 85.00% 28.55s [0.625 0.975 0.95 0.875 0.9 0.85]
% P2S3, ROIs 50, nbayes 87.92% 28.93s  [0.6 0.975 0.95 0.9 0.925 0.825]
% P2S3, ROIs 100, nbayes 87.08% 28.81s [0.625 0.975 0.95 0.875 0.925 0.85]

% S2P3, ROIs 30, nbayes 94.583% 29.1982s [0.8 1 1 0.9 1 0.975]
% S2P3, ROIs 50, nbayes 96.25% 29.5206s   [0.925 0.975 1 0.925 1 0.95]
% S2P3, ROIs 100, nbayes 95.00% 29.5154s [0.825 0.975 1 0.975 0.95 0.95]