% Example 4.4.1
% "Introduction to Pattern Recognition: A MATLAB Approach"
% S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras

close('all');
clear;

% 1. Generate vectors x1 and x2
randn('seed',0)
m1=8.75;
m2=9;
stdevi=sqrt(16);
N=1000;
x1=m1+stdevi*randn(1,N);
x2=m2+stdevi*randn(1,N);

x11=[x1 ; x1]';
x22=[x2 ; x2]';

% 2. Apply the t-test. Use MATLAB ttest2 function: two samples t-test

% If x and y are specified as matrices, they must have the same number 
% of columns. ttest2 performs a separate t-test along each column 
% and returns a vector of results.
rho=0.05
[h] = ttest2(x1,x2,rho)

% 3. Repeat with rho=0.001
rho=0.001
[h2] = ttest2(x11,x22,rho)