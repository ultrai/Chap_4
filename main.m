clear
d = pwd;
load('srinivasan.mat')
close all

n = 8
Feat_Train = Feat_train(Sub_idx_train<=n,:);
Label_Train = Label_train(Sub_idx_train<=n);
Feat_Test= Feat_test(Sub_idx_test>n,:);
Label_Test = Label_test(Sub_idx_test>n);

% Feat_Train_mean = mean(Feat_Train);
% Feat_Train = Feat_Train-ones(size(Feat_Train,1),1)*Feat_Train_mean;
% Feat_Train_std = std(Feat_Train);
% Feat_Train = Feat_Train./(ones(size(Feat_Train,1),1)*Feat_Train_std+0.01);
% Feat_Test = Feat_Test-ones(size(Feat_Test,1),1)*Feat_Train_mean;
% Feat_Test = Feat_Test./(ones(size(Feat_Test,1),1)*Feat_Train_std+0.01);

mi = FFCM_display(Feat_Train,Label_Train); %number of clusters to be 3
idx2=[];
for temp=1:max(Label_Train)
     idx2 = [idx2 min(pdist2(Feat_Test,mi{temp}),[],2)];
end
[temp,Est] = min(idx2,[],2);


ACC = mean(Est==Label_Test)

Decision = [];
for sub_test = (n+1):15
%     Feat_Test= Feat_test(Sub_idx_test==sub_test,:);
    Label_Test = Label_test(Sub_idx_test==sub_test);
    est = Est(Sub_idx_test(Sub_idx_test>n)==sub_test);
    est_1 = (mode(est(Label_Test==1))==1);
    est_2 = (mode(est(Label_Test==2))==2);
    est_3 = (mode(est(Label_Test==3))==3);
   Decision =[Decision; [est_1 est_2 est_3]];
end
