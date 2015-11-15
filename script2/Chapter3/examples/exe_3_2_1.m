% Exercise 3.2.1
% "Introduction to Pattern Recognition: A MATLAB Approach"
% S. Theodoridis, A. Pikrakis, K. Koutroumbas, D. Cavouras

close all
close('all');
clear;

% 1(a). To generate the data set X1 and a vector y1, whose i-th coordinate
% contains the class label of the i-th vector of X1, type
% randn('seed',0)  
% S=[.3 1. 1.; 1. 9. 1.; 1. 1. 9.];
% [l,l]=size(S);
% %mv=[-6 6 6; 6 6 6]';
% % with question 2, change the valus of mv as below:
% mv = [-6 6 6; 6 6 6]';
% % mv = [-2 0 0; 2 0 0]';
% N=200;
% X1=[mvnrnd(mv(:,1),S,N); mvnrnd(mv(:,2),S,N)]';
% y1=[ones(1,N), 2*ones(1,N)];
% 
% 
% % 1(b). To compute the eigenvalues/eigenvectors and variance percentages required in this step type
% m=3;
% [eigenval,eigenvec,explained,Y,mean_vec]=pca_fun(X1,m);
% eigenval
% explained
% 1(c). The projections of the data points of X1 along the direction of the first
% principal component are contained in the first row of Y , returned by the
% function pca_fun above. 

randn('seed',0)  
S=[.3 1. 1.; 1. 9. 1.; 1. 1. 9.];
[l,l]=size(S);

 mv = [-6 6 6; 6 6 6]';
% mv = [-2 0 0; 2 0 0]';
N=200;
X1=[mvnrnd(mv(:,1),S,N); mvnrnd(mv(:,2),S,N)]';
y1=[ones(1,N), 2*ones(1,N)];

% To compute the eigenvalues/eigenvectors and variance percentages required in this step type
m=3;
[eigenval,eigenvec,explained,Y,mean_vec]=pca_fun(X1,m);

% Compute the projections of X1
w1=eigenvec(:,1);
w2=eigenvec(:,2);

w12=cross(w1,w2);
t1=w12'*X1(:,y1==1);
t2=w12'*X1(:,y1==2);

X_proj12_1=X1(:,y1==1) - [t1;t1;t1].*((w12/(w12'*w12))*ones(1,length(t1)));
X_proj12_2=X1(:,y1==2) - [t2;t2;t2].*((w12/(w12'*w12))*ones(1,length(t2)));

% Plot the projections
% figure(1), plot(X1(1,y1==1),X1(2,y1==1),'r.',...
%     X1(1,y1==2),X1(2,y1==2),'bo')
% title('X11-X12')
% figure(2), plot(X1(1,y1==1),X1(3,y1==1),'r.',...
%     X1(1,y1==2),X1(3,y1==2),'bo')
% title('X11-X13')
% figure(3), plot(X1(2,y1==1),X1(3,y1==1),'r.',...
%     X1(2,y1==2),X1(3,y1==2),'bo')
% title('X12-X13')
% 
% figure(4), plot(Y(1,y1==1),Y(2,y1==1),'r.',...
%     Y(1,y1==2),Y(2,y1==2),'bo')
% title('Y1-Y2')
% figure(5), plot(Y(1,y1==1),Y(3,y1==1),'r.',...
%     Y(1,y1==2),Y(3,y1==2),'bo')
% title('Y1-Y3')
% figure(6), plot(Y(2,y1==1),Y(3,y1==1),'r.',...
%     Y(2,y1==2),Y(3,y1==2),'bo')
% title('Y2-Y3')
% 
% figure(7), plot3(X1(1,y1==1),X1(2,y1==1),X1(3,y1==1),'r.',...
%     X1(1,y1==2),X1(2,y1==2),X1(3,y1==2),'bo')
% 
% figure(8), plot3(X1(1,y1==1),X1(2,y1==1),X1(3,y1==1),'r.',...
%     X1(1,y1==2),X1(2,y1==2),X1(3,y1==2),'bo')
% figure(8), hold on
% figure(8), plot3(X_proj12_1(1,:),X_proj12_1(2,:),X_proj12_1(3,:),'k.',...
%     X_proj12_2(1,:),X_proj12_2(2,:),X_proj12_2(3,:),'ko')
% figure(8), axis equal




