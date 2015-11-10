function [info1,data1,meta1]=avgROIVoxelCond(info,data,meta, ROIs)
trials=find([info.cond]>1);
[info1,data1,meta1] = transformIDM_selectTrials(info,data,meta,trials);
[info1,data1,meta1] = createColToROI(info1,data1,meta1);
[info1,data1,meta1] = transformIDM_avgROIVoxels(info1,data1,meta1,ROIs);