load data-starplus-04799-v7;
data1 = data; meta1 = meta; info1 = info;
load data-starplus-04820-v7;
data2 = data; meta2 = meta; info2 = info;
load data-starplus-05675-v7;
data3 = data; meta3 = meta; info3 = info;
[rinfo,rdata,rmeta]=transformIDM_mergeMulti(info1,data1,meta1,info2,data2,meta2,info3,data3,meta3);