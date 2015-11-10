clear;
opts.lambda=[1 0 0 1];
opts.maxiter=1000;
X=rand(10,20, 30);
[Core, FACT] = HONMF(X, [3 3 3], opts);
Rec = honmf_rec(Core,FACT);