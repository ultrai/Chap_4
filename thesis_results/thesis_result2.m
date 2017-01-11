%function student()
clear
close all
%cd('/home/mict/for mam')
addpath(genpath('y_fcmc_ver.1.0'))
data = twospirals();
scatter(data(:,1), data(:,2), 12, data(:,3));
data(:,3) = data(:,3)+1;
Acc = [];
Train_Idx = [1:2:size(data,1)];
Test_Idx = [2:2:size(data,1)];
Feat_Train = data(Train_Idx,1:2);
Label_Train = data(Train_Idx,3);

Feat_Test = data(Test_Idx,1:2);
Label_Test = data(Test_Idx,3);
Feat_Train_mean = mean(Feat_Train);
Feat_Train = Feat_Train-ones(size(Feat_Train,1),1)*Feat_Train_mean;
Feat_Train_std = std(Feat_Train);
Feat_Train = Feat_Train./(ones(size(Feat_Train,1),1)*Feat_Train_std);
Feat_Test = Feat_Test-ones(size(Feat_Test,1),1)*Feat_Train_mean;
Feat_Test = Feat_Test./(ones(size(Feat_Test,1),1)*Feat_Train_std);

% close all
% hold on
% scatter(Feat_Train(Label_Train==1,1),Feat_Train(Label_Train==1,2),[],'r','o')
% scatter(Feat_Train(Label_Train==2,1),Feat_Train(Label_Train==2,2),[],'b','d')
% hold off    

mi = FFCM_display(Feat_Train,Label_Train);
