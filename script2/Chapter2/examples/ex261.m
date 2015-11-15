% exercise 2.6.1
% Vu Tien Duong

close('all');
clear

format compact
global figt4

% 1. Generate X1
l=2; %Dimensionality
poi_per_square=30; %Points per square
N=9*poi_per_square; %Total no. of points
rand('seed',0)
X1=[];
y1=[];
for i=0:2
    for j=0:2
        X1=[X1 rand(l,poi_per_square)+...
            [i j]'*ones(1,poi_per_square)];
        if(mod(i+j,2)==0)
            y1=[y1 ones(1,poi_per_square)];
        else
            y1=[y1 -ones(1,poi_per_square)];
        end
    end
end

% Plot X1
figure(1), plot(X1(1,y1==1),X1(2,y1==1),'r.',X1(1,y1==-1),X1(2,y1==-1),'bo')
figure(1), axis equal

% Generate X2
rand('seed',100)
X2=[];
y2=[];
for i=0:2
    for j=0:2
        X2=[X2 rand(l,poi_per_square)+...
            [i j]'*ones(1,poi_per_square)];
        if(mod(i+j,2)==0)
            y2=[y2 ones(1,poi_per_square)];
        else
            y2=[y2 -ones(1,poi_per_square)];
        end
    end
end

% Run the kernel perceptron algorithm for the linear kernel
% kernel='linear';
% kpar1=0;
% kpar2=0;
% max_iter=30000;
% [a,iter,count_misclas]=kernel_perce(X1,y1,kernel,kpar1,kpar2,max_iter);

% Run the algorithm using the radial basis kernel function
% kernel='rbf';
% kpar1=20;
% kpar2=0;
% max_iter=30000;
% [a,iter,count_misclas]=kernel_perce(X1,y1,kernel,kpar1,kpar2,max_iter);

% Run the algorithm using the polynomial kernel function
kernel='poly';
kpar1=1;
kpar2=15;
max_iter=30000;
[a,iter,count_misclas]=kernel_perce(X1,y1,kernel,kpar1,kpar2,max_iter);

% Compute the training error
for i=1:N
    K=CalcKernel(X1',X1(:,i)',kernel,kpar1,kpar2)';
    out_train(i)=sum((a.*y1).*K)+sum(a.*y1);
end
err_train=sum(out_train.*y1<0)/length(y1)
% where N is the number of training vectors.

% Compute the test error
for i=1:N
    K=CalcKernel(X1',X2(:,i)',kernel,kpar1,kpar2)';
    out_test(i)=sum((a.*y1).*K)+sum(a.*y1);
end
err_test=sum(out_test.*y2<0)/length(y2)

% Count the number of training vectors 
sum_pos_a=sum(a>0)

% 2. Plot the training set (see book Figures 2.7 and 2.8)
figure(1), hold on
figure(1), plot(X1(1,y1==1),X1(2,y1==1),'ro',...
    X1(1,y1==-1),X1(2,y1==-1),'b+')
figure(1), axis equal
% Note that the vectors of the training set from class 1 (?1) are marked by
% "o” (“+”). 

% Plot the decision boundary in the same figure
bou_x=[0 3];
bou_y=[0 3];
resolu=.05;
fig_num=1;
plot_kernel_perce_reg(X1,y1,a,kernel,kpar1,kpar2,bou_x,bou_y, resolu,fig_num)