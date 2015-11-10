function [Core, FACT] = decompose_tensor_HONMF(X, d)
opts.lambda=[1 0 0 1];
opts.maxiter=200; %1000
num_examples = size(X,1);
for j=1:num_examples
	X2 = X{j,1};
	[Core2, FACT2] = HONMF(X2, [3 3 3], opts);
	Core{j} = Core2;
	FACT{j} = FACT2;
end