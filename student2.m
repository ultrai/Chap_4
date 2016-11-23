%function student()
clear
close all
cd('/home/mict/Desktop/for mam')
run('results.m')
load('Matlab_data.mat');
exp =4;%1-4
load('Matlab_data.mat')
Acc = [];
data = data{exp};
data_Idx = isnan(sum(data));
data = data(:,data_Idx==0);

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



mi = FFCM(Feat_Train,Label_Train);
idx2=[];
for temp=1:max(Label_Train)
        idx2 = [idx2 min(pdist2(Feat_Test,mi{temp}),[],2)];
end
[temp,Idx2] = min(idx2,[],2);
Acc = [Acc;mean(Idx2==Label_Test)];
    
