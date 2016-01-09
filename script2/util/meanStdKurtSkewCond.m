function [examplesPS, examplesSS] =  meanStdKurtSkewCond(info,data,meta, ROIs, varargin)
examplesPS = [];
examplesSS = [];

for i=1:length(ROIs)
    l = length(varargin);
    ROI = ROIs(i);
    trials=find([info.cond]>1); 
    [info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);
    [info1,data1,meta1] = transformIDM_selectROIVoxels(info1,data1,meta1,ROI);
    if l==0
        [info1,data1,meta1] = avgVoxelSubset2(info1,data1,meta1);
    end
    
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
    
    examplesP = [examplesP2;examplesP3];
    examplesS = [examplesS2;examplesS3];
    
    if l > 0
        flag = 0;
        if l > 1 % use bias
            flag = 1;
        end
        tmean = mean(examplesP,2);
        tstd = std(examplesP,flag,2);
        tkurtosis = kurtosis(examplesP,flag,2);
        tskewness = skewness(examplesP,flag, 2);
        examplesP = [tmean tstd tkurtosis tskewness];
        
        tmean = mean(examplesS,2);
        tstd = std(examplesS,flag,2);
        tkurtosis = kurtosis(examplesS,flag,2);
        tskewness = skewness(examplesS,flag, 2);
        examplesS = [tmean tstd tkurtosis tskewness];
    end

    % combine examples and create labels.  Label 'picture' 1, label 'sentence' 2.
    examplesPS= [examplesPS examplesP];
    examplesSS= [examplesSS examplesS];
end