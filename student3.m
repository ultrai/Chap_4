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

c1 = fitcsvm(Feat_Train,Label_Train,'KernelFunction','linear');
[pred1,scores1] = predict(c1,Feat_Test);
Acc = [Acc;mean(pred1==Label_Test)];
c2 = fitcsvm(Feat_Train,Label_Train,'KernelFunction','rbf');
[pred2,scores2] = predict(c2,Feat_Test);
Acc = [Acc;mean(pred2==Label_Test)];
mi = FFCM_display(Feat_Train,Label_Train); %nc = 4; m=1.2;
idx2=[];
for temp=1:max(Label_Train)
        idx2 = [idx2 min(pdist2(Feat_Test,mi{temp}),[],2)];
end
[temp,Idx2] = min(idx2,[],2);
Col = [Idx2==1 Idx2==2 Idx2==3];
Acc = [Acc;mean(Idx2==Label_Test)];
%%
x = linspace(min([Feat_Train(:,1);Feat_Test(:,1)]),max([Feat_Train(:,1);Feat_Test(:,1)]));
y = linspace(min([Feat_Train(:,2);Feat_Test(:,2)]),max([Feat_Train(:,2);Feat_Test(:,2)]));
[X,Y] = meshgrid(x,y);


% [~,PosteriorRegion] = predict(c2,[X(:),Y(:)]);
% figure;
% contourf(X,Y,reshape(max(PosteriorRegion,[],2),size(X,1),size(X,2)),15);


% h = colorbar;
% h.YLabel.String = 'Maximum posterior';
% h.YLabel.FontSize = 15;
% hold on
% gh = gscatter(X(:,1),X(:,2),Y,'krk','*xd',8);
% gh(2).LineWidth = 2;
% gh(3).LineWidth = 2;
% title 'Iris Petal Measurements and Maximum Posterior';
% xlabel 'Petal length (cm)';
% ylabel 'Petal width (cm)';
% axis tight
% legend(gh,'Location','NorthWest')
% hold off
