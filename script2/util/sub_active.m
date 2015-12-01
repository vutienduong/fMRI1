function activeVoxels = sub_active(varargin)
  
  l = length(varargin);
  
  if l < 1
    fprintf('syntax: transformIDM_selectActiveVoxact(IDM,<# to keep>,[[list of conditions to t-test]])\n');
    return;
  end
    
  info = varargin{1};
  data = varargin{2};
  meta = varargin{3};
  nToKeep = varargin{4};

  % gather information about number,type and length of trials and other dataset parameters
  %[ntrials,nvoxels,nconds,minTrialLenCond,ntrialsCond,trialBegin,trialEnd] = IDMinformation( info,data,meta,meta.study );
  IDM_information=IDMinformation( info,data,meta,meta.study );
  ntrials=IDM_information.nTrials;
  nvoxels=IDM_information.nVoxels;
  nconds=IDM_information.nConds;
  minTrialLenCond=IDM_information.minTrialLenCond;
  ntrialsCond=IDM_information.nTrialsCond;
  
  
  if l > 4
    % user wants to specify conditions
    conditions = varargin{5};
  else
    conditions = 2:nconds;% use all of them
  end
  
  if nToKeep > nvoxels
    if nvoxels
      fprintf('warning: roi %s has %d voxels and you asked me to keep %d.\n',meta.roi,nvoxels,nToKeep);
      nToKeep = nvoxels;
    else
      fprintf('ERROR: roi %s does not contain voxels\n',meta.roi);
      return;
    end    
  end
  
  fprintf('transformIDM_selectActiveVoxact: keeping %d voxels from conditions ',nToKeep);
  fprintf('%d ',conditions); fprintf('\n');
  
  
  information = localInformation(meta.study);
  dimensions  = information.dimensions;
  dimx        = dimensions(1); dimy = dimensions(2); dimz = dimensions(3);
  
  % finds the top <nToKeep> most active voxels in each condition
  % and keeps all of those - at most #ROIS*#conditions*#nToKeep voxels
  % will be kept
  nvoxels = size(data{1},2); % WARNING: fails if data{1}==[]
  activeCount = zeros(1,nvoxels);

  % get the Tvalues
  if 1
    [results] = mri_computeTvalues2 (info,data,meta);
    %    save carago.mat results;
  else
    %    load carago.mat
  end
    
  nconds    = size(results,1);
    
  % Create a ranking of the voxels by p-value of the activity
  % test of condition <c> against fixation.
  % Sorted in increasing order (small pvalue -> more active) 
  for i=1:length(conditions)
    c = conditions(i);
    % DUONG
    [sortedPValues{c},sortedVoxels{c}] = sort(results{1,c}, 'descend');
    idx(c) = 1; % keeps track of selected from this condition
  end

  % Conceptually, this works by taking the per condition rankings
  % Cond  2   ... last condition being considered
  %       v(1)    v(1)
  %       v(2)    v(2)
  % and  -------------- sliding a window down until the number left
  % above is the # to keep.
  %
  % Note that some voxels may be near the top of several of the rankings.
  
  % keeps count of how often each voxel is active
  activeCount = zeros(1,nvoxels);

  % how many to aggregate per round
  nLeft    = nToKeep;  
  nPerCond = floor( nLeft/length(conditions) );

  while nLeft > 0
%    fprintf('nLeft=%d nPerCond=%d\n',nLeft,nPerCond);

    for i=1:length(conditions)
      c = conditions(i);
      indices = sortedVoxels{c}(idx(c):(idx(c)+nPerCond-1));
%      fprintf('\tc=%d\t',c); fprintf('%d ',indices);fprintf('\n');
      activeCount(indices) = activeCount(indices) + 1;      
      idx(c) = idx(c) + nPerCond;

      nLeft = nToKeep - sum(activeCount>0);
      if nLeft <= 0; break; end
    end    

%    fprintf('prior nLeft=%d\n',nLeft);
%    disp(find(activeCount > 0));    
%    fprintf('after nLeft=%d\n',nLeft);
%    pause
    
    if nLeft <= 0; break; end
    nPerCond = floor(nLeft/length(conditions));
  
    if nPerCond == 0; nPerCond = 1; end
  end
  
  activeVoxels = find(activeCount > 0); % these will be used