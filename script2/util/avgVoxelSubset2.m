function [info,avgdata,meta] = avgVoxelSubset2 ( info,data,meta )
ntrials = max(size(info));
%nvoxels = size(data{1},2);
avgdata = cell(ntrials,1);

for i=1:1:ntrials
  % average of all cols, without weight matrix
  % spare copying of everything
  tmean = mean(data{i},2);
  tstd = std(data{i},0,2);
  tkurtosis = kurtosis(data{i},0,2);
  tskewness = skewness(data{i},0, 2);
  avgdata{i} = [tmean tstd tkurtosis tskewness];
end