function [param,idx_out]=transvm_train(X_train, Y_train, idx_feat,extra_param)



% C      = 0.1;
% kernel = polynomial(2);
% kernel = linear;
C = extra_param.C;
kernel = extra_param.kernel;
X_extra = extra_param.X_extra;
%stages = extra_param.stages;
stages = extra_param.trans_num % number of bootstrap stages
perc = extra_param.perc; % at each stage add the "perc" of the point wuth the largest margin 
tutor  = smosvctutor;

% train support vector machine

fprintf(1,'training support vector machine...\n');

% first train - based on the labeled data only
zeta=Y_train;
ratio=sum(Y_train>0)/sum(Y_train<0);
I=find(zeta<0);
zeta(I)=ratio;
net = train(svc, tutor, X_train(:,idx_feat), Y_train, C, kernel,zeta); 

for stg = 1:stages,
    disp(['stage ' num2str(stg) ' of ' num2str(stages) ' stages']);
    Y_conf = fwd(net, X_extra(:,idx_feat));
    Y_resu = sign(Y_conf);
    Y_conf = abs(Y_conf);
    [s I] = sort(Y_conf);
    num = floor( length(Y_resu) * perc);
    add_idx = I((end-num+1):end);
    X_train = [X_train; X_extra(add_idx,:)];
    Y_train = [Y_train; Y_resu(add_idx)];
    rm_idx = setdiff(1:size(X_extra,1), add_idx);
    X_extra = X_extra(rm_idx,:);

    zeta=Y_train;
    ratio=sum(Y_train>0)/sum(Y_train<0)
    I=find(zeta<0);
    zeta(I)=ratio;
    net = train(svc, tutor, X_train(:,idx_feat), Y_train, C, kernel,zeta);
end

param = net;
idx_out=idx_feat;
