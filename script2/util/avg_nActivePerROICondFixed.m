function [examplesPS, examplesSS] =  avg_nActivePerROICondFixed(info,data,meta, ROIs, num_per_ROI)
examplesPS = [];
examplesSS = [];

for i=1:length(ROIs)
    ROI = ROIs(i);
    trials=find([info.cond]>1); 
    [info2,data2,meta2] = transformIDM_selectTrials(info,data,meta,trials);
    [info2,data2,meta2] = transformIDM_selectROIVoxels(info2,data2,meta2,ROI);
    [info2,data2,meta2] = transformIDM_selectActiveVoxact(info2,data2,meta2,num_per_ROI);
    [info1,data1,meta1] = transformIDM_avgVoxelSubset(info2,data2,meta2);
    
    % collect the non-noise and non-fixation trials
    [infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
    [infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));
 
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
    examplesPS= [examplesPS [examplesP2;examplesP3]];
    examplesSS= [examplesSS [examplesS2;examplesS3]];
end