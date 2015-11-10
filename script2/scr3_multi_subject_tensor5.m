% tinh FDR cho ca train va test cung luc
% vars for global (whole program)
t = cputime;
half_examples = size(Core,2)/2;
examples = [];
num_subjects = half_examples/2;

% FALSE: use all features
use_FDR = true;

% FALSE: use leave one example
use_L1Subject = false;

% create labels
labelsP = ones(half_examples,1) + 1;
labelsS = ones(half_examples,1);
labels = [labelsP;labelsS];

% length of vector
vector_dim = prod(size(Core{1,1}));

% vectorized from tensor form (examples <- Core)
for i=1:half_examples*2
    coreTemp = Core{1,i};
    examples(i,:) = reshape(coreTemp, 1, vector_dim);
end

% create examples from P, S
examplesP = examples(1:half_examples,:);
examplesS = examples(half_examples + 1 : half_examples * 2, :);

% number of features
numfeat = size(examplesP,2);

if use_FDR
    %%%%%%%%%%%%%%%%%%%%%%%%%% FDR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % calculate FDR
    for i=1:numfeat
        fdr(i)= Fisher(examplesP(:,i),examplesS(:,i));
    end

    % ranked using FDR
    [fdr,featrank]=sort(fdr,'descend');
    examplesPR = examplesP(:,featrank); 
    examplesSR = examplesS(:,featrank);

    % vars for classifier
    avg_acc = [];
    select_num = 100;

    % iterate through many settings of the number of selected features (Ex.
    % from 1 to 20)
    for numOfFeature = 1:select_num
        % get features using index belonged to the highest FDR
        examplesPS = examplesPR(:,1:numOfFeature); 
        examplesSS = examplesSR(:,1:numOfFeature);
        examples = [examplesPS; examplesSS];

        % var of accuracy
        adb_acc = [];

        % iterate through all subjects, each test use n-1 for training and 1
        % for testing
        if use_L1Subject %(L1Subject)===
            % examples = 
            %   P2 (6 rows)
            %   P3 (6 rows)
            %   S2 (6 rows)
            %   S3 (6 rows)
            
            for i=1:num_subjects
                % index of test, train samples
                tridx = ones(half_examples*2,1);
                tridx(num_subjects * [1:4] - num_subjects + i,:) = 0;
                tridx = logical(tridx);
                teidx = not(tridx);
                
                % accuracy
                adb_acc(1,i) = util_classifier(tridx, teidx, examples, labels);
            end
            % average of all folds
            avg_acc(numOfFeature) = sum(adb_acc)/num_subjects;
            
        else %(L1Example)=============
            for i=1:half_examples*2
                % index of test, train samples
                tridx = ones(half_examples*2,1);
                tridx(i,1) = 0;
                tridx = logical(tridx);
                teidx = not(tridx);

                % accuracy
                adb_acc(1,i) = util_classifier(tridx, teidx, examples, labels);
            end
            % average of all folds
            avg_acc(numOfFeature) = sum(adb_acc)/(half_examples*2);
        end % L1Example===============
    end

    % plot the graph of the settings and the accuracy
    plot(1:select_num, avg_acc);

    % processing time
    e = cputime - t;
    disp(['accuracy ', num2str(avg_acc(9)),num2str(avg_acc(3)),num2str(avg_acc(4)) , ' | processing time ', num2str(e)]);
    %%%%%%%%%%%%%%%%%% end FDR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    %%%%%%%%%%%%%%%%%%% all feature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % vars for classifier
    avg_acc = [];
    adb_acc = [];

    % iterate through all subjects, each test use n-1 for training and 1
    % for testing
    if use_L1Subject %(L1Subject)===
        for i=1:num_subjects
            % index of test, train samples (*)
            tridx = ones(half_examples*2,1);
            tridx(num_subjects * [1:4] - num_subjects + i,:) = 0;
            tridx = logical(tridx);
            teidx = not(tridx);

            % accuracy
            adb_acc(1,i) = util_classifier(tridx, teidx, examples, labels);
        end
        % average of all folds
        avg_acc = sum(adb_acc)/num_subjects;
    else %(L1Example)================
        for i=1:half_examples*2
            % index of test, train samples (*)
            tridx = ones(half_examples*2,1);
            tridx(i,1) = 0;
            tridx = logical(tridx);
            teidx = not(tridx);
            
            % accuracy
            adb_acc(1,i) = util_classifier(tridx, teidx, examples, labels);
        end
        % average of all folds
        avg_acc = sum(adb_acc)/(half_examples*2);
    end % L1Example===================
    
    % processing time
    e = cputime - t;
    disp(['accuracy ', num2str(avg_acc) , ' | processing time ', num2str(e)]);
    %%%%%%%%%%%%%%%%%%% end all feature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end











% RESULT
% use_FDR      use_L1Subject     acc
% false        true              54.167
% false        false             54.167