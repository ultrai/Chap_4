clear
d = pwd;
load('srinivasan.mat')
%%
n = 15;
Val = [];
Decision=[];
Decision2=[];
for n =1:15
    n
    Feat_Train = Feat_train(Sub_idx_train~=n,:);
    Feat_Test= Feat_test(Sub_idx_test==n,:);
    Label_Train = Label_train(Sub_idx_train~=n);
    Label_Test= Label_test(Sub_idx_test==n);
    mi = FFCM_display(Feat_Train,Label_Train);
    idx2=[];
    for temp=1:max(Label_Train)
        idx2 = [idx2 min(pdist2(Feat_Test,mi{temp}),[],2)];
    end
    [temp,est] = min(idx2,[],2);
    %Val = [Val;sum(est==Label_Test)/numel(est)];
    est_1 = (mode(est(Label_Test==1))==1);
    est_2 = (mode(est(Label_Test==2))==2);
    est_3 = (mode(est(Label_Test==3))==3);
    Decision2 =[Decision2; [est_1 est_2 est_3]];
end
%%
Decision2=[];
for cv =1:15
    cv
    n=7;
    list = [1:15]';
    list = circshift(list,cv-1);
    Idx_train = zeros(size(Sub_idx_train,1),1);
    Idx_test = zeros(size(Sub_idx_test,1),1);
    for temp = 1:(15-n)
        Idx_train(Sub_idx_train==list(temp)) = temp;
    end
    for temp = (15-n+1):15
        Idx_test(Sub_idx_test==list(temp)) = temp;
    end
    
    Feat_Train = Feat_train(Idx_train>0,:);
    Feat_Test= Feat_test(Idx_test>0,:);
    Label_Train = Label_train(Idx_train>0);
    %Label_Test= Label_test(abs(Sub_idx_test-cv+1)<=n);
    mi = FFCM_display(Feat_Train,Label_Train);

    idx2=[];
    for temp=1:max(Label_Train)
        idx2 = [idx2 min(pdist2(Feat_Test,mi{temp}),[],2)];
    end
    [temp,Est] = min(idx2,[],2);
    
    for sub_test = (15-n+1):15
%     Feat_Test= Feat_test(Sub_idx_test==sub_test,:);
        Label_Test = Label_test(Idx_test==sub_test);
        est = Est(Idx_test(Idx_test>n)==sub_test);
        est_1 = (mode(est(Label_Test==1))==1);
        est_2 = (mode(est(Label_Test==2))==2);
        est_3 = (mode(est(Label_Test==3))==3);
        Decision2 =[Decision2; [est_1 est_2 est_3]];
    end
end
mean(Decision2)
%bench 0.9429    0.8857    0.9143
%nc=2 0.9905    0.8476    0.9333
%nc=3 1.0000    0.8667    0.9143
%nc=3 m=1.1 1.0000    0.8762    0.9524


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
