close('all');
clear

N1= 50;
N2= 50;

randn('seed' ,0)
m1 = [1 1]; S1 = eye(2);
X1 = mvnrnd(m1 ,S1, N1 )' ;

randn('seed' ,0)
m2 = [4 4]; S2 = eye(2);
X2 = mvnrnd(m2 ,S2, N2 )' ;

X=[X1 X2];
y=[ones(1,N1) -ones(1,N2)];

figure(1), hold on
figure(1), plot(X(1,y==1),X(2,y==1),'r.',X(1,y==-1),X(2,y==-1),'bx')

% 1.
T_max=3000; % max number of base classifiers
[pos_tot, thres_tot, sleft_tot, a_tot, P_tot,K] = boost_clas_coord(X, y, T_max);

% 2.
[y_out, P_error] =boost_clas_coord_out(pos_tot, thres_tot, sleft_tot, a_tot, P_tot,K,X, y);
figure(2), plot(P_error)

% % 3. Test set
% randn('seed' ,100)
% Z1 = mvnrnd(m1 ,S1, N1 )' ;

% randn('seed' ,100)
% Z2 = mvnrnd(m2 ,S2, N2 )' ;

% Z=[Z1 Z2];
% y=[ones(1,N1) -ones(1,N2)];

% % Classify the vectors of Z
% [y_out_Z, P_error_Z] =boost_clas_coord_out(pos_tot, thres_tot, sleft_tot, a_tot, P_tot,K,Z, y);
% figure(3), plot(P_error_Z)