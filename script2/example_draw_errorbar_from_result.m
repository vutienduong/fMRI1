load result
irisMeans = mean(result);
irisSTDs  = std(result);

figure
errorbar(irisMeans, irisSTDs, 's')

% Add title and axis labels
title('Comparison of three species in the Fisher Iris data')
xlabel('Species of Iris')
ylabel('Mean size in mm')
box on

% Change the labels for the tick marks on the x-axis
irisSpecies = {'ROIs', 'Active', 'iPCA', 'FDR', 'proposed'};
set(gca, 'XTick', 1:3, 'XTickLabel', irisSpecies)

% Create labels for the legend
%irisMeas = {'Sepal length', 'Sepal width', 'Petal length', 'Petal width'};
%legend(irisMeas, 'Location', 'Northwest')