% for each person , length(file_name)
clear; 
file_name = {'data-starplus-04847-v7', 'data-starplus-04799-v7', 'data-starplus-05710-v7',...
    'data-starplus-04820-v7', 'data-starplus-05675-v7', 'data-starplus-05680-v7'};
    nrois = 25;
    nsubjects = 6;
matr = zeros(nsubjects, nrois);
for j=1:6
    load(file_name{j});
    disp(['Complete load : ', file_name{j}]);
    for k=1:25
        matr(j,k) = size(meta.rois(k).columns, 2);
    end
end
