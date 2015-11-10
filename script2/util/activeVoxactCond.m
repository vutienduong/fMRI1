function [info1,data1,meta1]=activeVoxactCond(info,data,meta, ROIs, num_selected_feature)
trials=find([info.cond]>1); 
[info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
[info1,data1,meta1]=transformIDM_selectROIVoxels(info1,data1,meta1,ROIs);
[info1,data1,meta1] = transformIDM_selectActiveVoxact(info1,data1,meta1,num_selected_feature);
% trials=find([info1.cond]>1); 
% [info1,data1,meta1]=transformIDM_selectTrials(info1,data1,meta1,trials);