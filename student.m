%function student()
clear
close all
%cd('/home/mict/for mam')
load('mam_data.mat');
addpath(genpath('fcmc'))
exp =3;%1-3
Acc = [];
Train_Idx = [1:2:15];
Test_Idx = [2:2:16];
Feat_Train = data(Train_Idx,1:2);
Label_Train = data(Train_Idx,exp+2);
Feat_Test = data(Test_Idx,1:2);
Label_Test = data(Test_Idx,exp+2);
Feat_Train_mean = mean(Feat_Train);
Feat_Train = Feat_Train-ones(size(Feat_Train,1),1)*Feat_Train_mean;
Feat_Train_std = std(Feat_Train);
Feat_Train = Feat_Train./(ones(size(Feat_Train,1),1)*Feat_Train_std);
Feat_Test = Feat_Test-ones(size(Feat_Test,1),1)*Feat_Train_mean;
Feat_Test = Feat_Test./(ones(size(Feat_Test,1),1)*Feat_Train_std);

init_V = [6.0,1379.0;5,817];
nc = 2; m=1.5; max_iter=100; term_thr=1e-15; eta = 2.0;
[mi_s,U,E] = Yf_FCMC1([Feat_Train;Feat_Test],nc, [m; max_iter; term_thr;  1; 1], init_V);
[temp,Idx ]= max(U(:,numel(Train_Idx)+1:end));
% m_i = zeros(nc,1);
% for clust_Id = 1:nc
%     [temp,m_i(clust_Id)] = min(pdist2(mi_s,mean(Feat_Train(Label_Train==clust_Id,:))));
% end
%m_i(Idx)
% 
if exp==1
    Acc = [Acc;mean((3-Idx)'==Label_Test)];
end
if exp==2
    Acc = [Acc;mean((3-Idx)'==Label_Test)];
end
if exp==3
    Acc = [Acc;mean((Idx)'==Label_Test)];
end

c1 = fitcsvm(Feat_Train,Label_Train,'KernelFunction','linear');
[pred1,scores1] = predict(c1,Feat_Test);
Acc = [Acc;mean(pred1==Label_Test)];
c2 = fitcsvm(Feat_Train,Label_Train,'KernelFunction','rbf');
[pred2,scores2] = predict(c2,Feat_Test);
Acc = [Acc;mean(pred2==Label_Test)];
mi = FFCM(Feat_Train,Label_Train);
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
%%
z1 = ones(size(X))*1e14;
z2 = z1;
n=2;
for temp=1:size(mi{1},1)
    z1 = min(((X-mi{1}(temp,1)).^n+(Y-mi{1}(temp,2)).^n).^(1/n),z1);
end
for temp=1:size(mi{2},1)
    z2 = min(((X-mi{2}(temp,1)).^n+(Y-mi{2}(temp,2)).^n).^(1/n),z2);
end
%Z = min(mat2gray(z1),mat2gray(z2));
%tt2=Z;

%Z = z1-z2;

mask = z1<z2;
Z = (max(z1(:))-z1).*mask-(max(z2(:))-z2).*(1-mask);
tt2=mat2gray(Z);
figure, hold on
contourf(X,Y,tt2/2,15);
for temp=1:1:size(mi{1},1)
    scatter(mi{1}(temp,1),mi{1}(temp,2),[],[0 0 0],'+')
end
for temp=1:1:size(mi{2},1)
    scatter(mi{2}(temp,1),mi{2}(temp,2),[],[0 0 0],'d')
end
Markers = ['o','s'];
for temp=1:2
        ydata_temp = Feat_Test(Label_Test==temp,:);
        scatter(ydata_temp(:,1),ydata_temp(:,2),[],Col(Label_Test==temp,:),Markers(temp))
    end
hold off
