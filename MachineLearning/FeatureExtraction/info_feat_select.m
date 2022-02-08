function idx_feat=angliainfo_feat_select(X_train, Y_train, extra_param)
    
feat_num = extra_param.feat_num;

% force the data to be binary
med = median(X_train);
X_train = X_train > med(ones(1,size(X_train,1)),:);

fmi = featureInformation(X_train, Y_train);

[sfmi Ifmi] = sort(fmi);

idx_feat = Ifmi((end-feat_num+1):end);

% idx_feat = idx_feat([1:7 8:15]); % REMOVE IT !

% idx_feat = idx_feat([50    49    48    47    46    45    44    43    41    40    38    36    32    21    14    13  9     8     4]); % REMOVE IT !