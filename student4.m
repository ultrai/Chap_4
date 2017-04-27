%function student()
clear
close all
%cd('/home/mict/Desktop/for mam')
addpath(genpath('thesis_results'))
load('breast_cancer_data.mat');
Acc = [];
data = A(:,2:end);
data(:,end) = (data(:,end)>3)+1;

Train_Idx = [1:2:size(data,1)-1];
Test_Idx = [2:2:size(data,1)];
data = sortrows(data,size(data,2));
Feat_Train = data(Train_Idx,1:size(data,2)-1);
Label_Train = data(Train_Idx,size(data,2));
Feat_Test = data(Test_Idx,1:size(data,2)-1);
Label_Test = data(Test_Idx,size(data,2));


Feat_Train_mean = mean(Feat_Train);
Feat_Train = Feat_Train-ones(size(Feat_Train,1),1)*Feat_Train_mean;
Feat_Train_std = std(Feat_Train);
Feat_Train = Feat_Train./(ones(size(Feat_Train,1),1)*Feat_Train_std+0.1);
Feat_Test = Feat_Test-ones(size(Feat_Test,1),1)*Feat_Train_mean;
Feat_Test = Feat_Train./(ones(size(Feat_Test,1),1)*Feat_Train_std+0.1);


% nc = max(Label_Train); m=1.1; max_iter=100; term_thr=1e-15; eta = 2.0;
% init_V = Feat_Train(1:nc,:);
% [mi_s,U,E] = Yf_FCMC1([Feat_Train;Feat_Test],nc, [m; max_iter; term_thr;  1; 1], init_V);
% [temp,Idx ]= max(U(:,numel(Train_Idx)+1:end));
% % m_i = zeros(nc,1);
% % for clust_Id = 1:nc
% %     [temp,m_i(clust_Id)] = min(pdist2(mi_s,mean(Feat_Train(Label_Train==clust_Id,:))));
% % end
% %m_i(Idx)

t = templateSVM('KernelFunction','linear');
Mdl = fitcecoc(Feat_Train,Label_Train,'Learners',t);
pred1 = predict(Mdl,Feat_Test);
Acc = [Acc;mean(pred1==Label_Test)];
t = templateSVM('KernelFunction','gaussian');
Mdl = fitcecoc(Feat_Train,Label_Train,'Learners',t);
pred1 = predict(Mdl,Feat_Test);
Acc = [Acc;mean(pred1==Label_Test)];
t = templateSVM('KernelFunction', 'rbf');
Mdl = fitcecoc(Feat_Train,Label_Train,'Learners',t);
pred1 = predict(Mdl,Feat_Test);
Acc = [Acc;mean(pred1==Label_Test)];



mi = FFCM_display(Feat_Train,Label_Train);%nc=9
X = [];
Y = [];
for iter = 1:numel(mi)
    X_temp = mi{iter};
    X = [X;X_temp];
    Y = [Y;zeros(size(X_temp,1),1)+iter];
end
Mdl = fitcknn(X,Y,'NumNeighbors',6,'Standardize',0);
Idx2 = predict(Mdl,Feat_Test);
%mean(Idx2==Label_Test)
Acc = [Acc;mean(Idx2==Label_Test)]
    
