index = 4;
sym1 = info1(index).sentenceSym1;
sym2 = info1(index).sentenceSym2;
rel = info1(index).sentenceRel;
ccat = strcat(sym1(1), rel(1), sym2(1), ' ');
img = info1(index).img;
img = img(1:3);
if strcmp(ccat, img)
    disp('aa TRUE');
else
    disp('aa FALSE');
end