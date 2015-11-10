function [info1,data1,meta1] =  nActivePerROICond(info,data,meta, ROIs, num_per_ROI)
for i=1:length(ROIs)
    ROI = ROIs(i);
    trials=find([info.cond]>0); 
    [info2,data2,meta2] = transformIDM_selectTrials(info,data,meta,trials);
    [info2,data2,meta2] = transformIDM_selectROIVoxels(info2,data2,meta2,ROI);
    [info2,data2,meta2] = transformIDM_selectActiveVoxels(info2,data2,meta2,num_per_ROI);
    [info2,data2,meta2] = transformIDM_avgVoxelSubset(info2,data2,meta2);
    if i>1
        [info1,data1,meta1]=transformIDM_mergeMulti(info1,data1,meta1,info2,data2,meta2);
    else
        info1 = info2; data1 = data2; meta1 = meta2;
    end
end
trials=find([info1.cond]>1); 
[info1,data1,meta1] = transformIDM_selectTrials(info1,data1,meta1,trials);