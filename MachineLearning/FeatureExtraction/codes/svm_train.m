function [param,idx_out]=svm_train(X_train, Y_train, idx_feat,extra_param)



% C      = 0.1;
% kernel = polynomial(2);
% kernel = linear;
C = extra_param.C;
kernel = extra_param.kernel;
tutor  = smosvctutor;

% train support vector machine

fprintf(1,'training support vector machine...\n');

zeta=Y_train;
ratio=sum(Y_train>0)/sum(Y_train<0);
I=find(zeta<0);
zeta(I)=ratio;
net = train(svc, tutor, X_train(:,idx_feat), Y_train, C, kernel,zeta); 

% net = train(svc, tutor, X_train(:,idx_feat), Y_train, C, kernel);

% net = fixduplicates(net, X_train(:,idx_feat), Y_train);

%net = strip(net);
param = net;
idx_out=idx_feat;

