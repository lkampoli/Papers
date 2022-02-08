function [idx_feat,extra_param,fMI] = infoSleeping_feat_select(X_train, Y_train, extra_param)
    


zc = 0.5; % zero correction

    
Iplus = (Y_train > 0);
Nplus = sum(Iplus);
Nminus = length(Y_train) - Nplus;
fpcount = sum(X_train(Iplus,:));
fncount = sum(X_train(~Iplus,:));
pos = (fpcount+0.5)/Nplus;
neg = (fncount+0.5)/Nminus;

fMI = (pos - neg).*log(pos./neg);


[s I] = sort(fMI);
I = I(length(I):-1:1);
idx_feat = I(1:extra_param.feat_num);
