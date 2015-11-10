function [i, d, m] = scr3_transformIDM(info, data, meta, transform_type)
	nKeeps = 20;
	switch(transform_type)
		case 'activeVoxact'
			[i,d,m, activeVoxels] = transformIDM_selectActiveVoxact(info,data,meta,nKeeps);
		case 'activeVoxel'
			[i,d,m] = transformIDM_selectActiveVoxels(info,data,meta,20);
		otherwise
			trials=find([info.cond]>1); 
		    [info1,data1,meta1]=transformIDM_selectTrials(info,data,meta,trials);
		    % seperate P1st and S1st trials
		    [info2,data2,meta2]=transformIDM_selectROIVoxels(info1,data1,meta1,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'});
		    
	end
