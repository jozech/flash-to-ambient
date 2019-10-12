clc;
clear all;

root = '../../datasets/DATASET_LR/test/';
flashFiles = dir('../../datasets/DATASET_LR/test/*flash.png');

root_results = '../../results/LIME/';

if ~exist(root_results, 'dir')
    mkdir(root_results);
end

for k=1:numel(flashFiles)
    display(flashFiles(k).name);
    filename = strcat(root,flashFiles(k).name);
    X = enhanceFn(filename);
    
    imwrite(X, strcat(root_results,flashFiles(k).name))
end

