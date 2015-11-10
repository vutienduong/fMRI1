function [Core, FACT] = decompose_tensor_Tucker(X,d)
% opts.lambda=[1 0 0 1];
% opts.maxiter=200; %1000
num_examples = size(X,1);
t = cputime;
for j=1:num_examples
	X2 = X{j,1};
    [FACT2,Core2,ExplX,Xm]=tucker(X2,d);
	Core{j} = Core2;
	FACT{j} = FACT2;
    
end
e = cputime - t;
disp(['processing time ', num2str(e)]);