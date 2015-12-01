load('example4indexs.mat');
examplesP = [];
examplesS = [];
for i=1:2:11
    indexP = (i*40-39) : i*40 ;
    examplesP = [examplesP; examples(indexP,:)]; 
end

for i=2:2:12
    indexS = (i*40-39) : i*40 ;
    examplesS = [examplesS; examples(indexS,:)]; 
end

examples = [examplesP; examplesS];