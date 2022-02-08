function [param,idx_out]=perceptron_train(X_train, Y_train, idx_feat,extra_param)


num = length(Y_train);
w = zeros(num,1);
K = zeros(num,num);

blocks=[1:1000:num num+1];

for ii = 1:(length(blocks)-1)
    for jj = 1:ii
         Ktemp = evaluate(extra_param.kernel,X_train(blocks(ii):(blocks(ii+1)-1),idx_feat),...
                                             X_train(blocks(jj):(blocks(jj+1)-1),idx_feat));
         K(blocks(ii):(blocks(ii+1)-1),blocks(jj):(blocks(jj+1)-1)) = Ktemp;
         K(blocks(jj):(blocks(jj+1)-1),blocks(ii):(blocks(ii+1)-1)) = Ktemp';
         disp([num2str(ii) ' ' num2str(jj)]);
     end
end
   

saf=extra_param.saf;

for itr = 1:extra_param.max_itr
    pred = K*w;
    misslabled = find(pred.*Y_train<=saf);
    disp(['iteration ' num2str(itr) ' misslabled ' num2str(length(misslabled)) ' max ' num2str(max(abs(w)))]);
    if (isempty(misslabled))
        break
    end
    for jj = 1:length(misslabled)
        ii = misslabled(jj);
        pred=K(ii,:)*w;
        if (pred*Y_train(ii)<=saf)
            w(ii)=w(ii)+Y_train(ii);
            flag=0;
        end
    end
    if (flag)
        break;
    end    
end

idx_out=idx_feat;
param.w=w;
param.kernel=extra_param.kernel;

