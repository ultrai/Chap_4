clear
d = pwd;
load('srinivasan.mat')
%%
%%

n = 8
Feat_Train = Feat_train(Sub_idx_train<=n,:);
Label_Train = Label_train(Sub_idx_train<=n);
Feat_Test= Feat_test(Sub_idx_test>n,:);
Label_Test = Label_test(Sub_idx_test>n);

% Feat_Train_mean = mean(Feat_Train);
% Feat_Train = Feat_Train-ones(size(Feat_Train,1),1)*Feat_Train_mean;
% Feat_Train_std = std(Feat_Train);
% Feat_Train = Feat_Train./(ones(size(Feat_Train,1),1)*Feat_Train_std);
% Feat_Test = Feat_Test-ones(size(Feat_Test,1),1)*Feat_Train_mean;
% Feat_Test = Feat_Test./(ones(size(Feat_Test,1),1)*Feat_Train_std);

mi = FFCM_display(Feat_Train,Label_Train);
