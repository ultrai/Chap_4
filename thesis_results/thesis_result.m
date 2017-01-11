clear
close all
%cd('/home/mict/for mam')
load('mam_data.mat');
addpath(genpath('fcmc'))
b = (data(:,3)==1).*(data(:,4)==2);
a = (data(:,3)==1).*(data(:,4)==1);
c = (data(:,3)==2).*(data(:,5)==1);

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


data(Train_Idx,1:2) = Feat_Train;
data(Test_Idx,1:2) = Feat_Test;

close all
h = figure,hold on

scatter(data(:,1),data(:,2),1,'.')
text(data(a==1,1),data(a==1,2),'a','FontSize',21);
text(data(b==1,1),data(b==1,2),'b','FontSize',21);
text(data(c==1,1),data(c==1,2),'c','FontSize',21);
set(gca,'fontsize',18)
%axis([ax1 ax2],[-2 3 -2 2])
hold off
saveas(h,['Toy_example.png'])
%%
mi = FFCM_display(Feat_Train,Label_Train);
idx2=[];
for temp=1:max(Label_Train)
        idx2 = [idx2 min(pdist2(Feat_Test,mi{temp}),[],2)];
end
[temp,Idx2] = min(idx2,[],2);
Col = [Idx2==2 Idx2==3 Idx2==1];
Acc = [Acc;mean(Idx2==Label_Test)];
%%
x = linspace(-2,3);
y = linspace(-2,2);
[X,Y] = meshgrid(x,y);

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
% close all
h = figure, hold on
contourf(X,Y,tt2/2,15);
for temp=1:1:size(mi{1},1)
    scatter(mi{1}(temp,1),mi{1}(temp,2),150,[0 0 1],'s')
end
for temp=1:1:size(mi{2},1)
    scatter(mi{2}(temp,1),mi{2}(temp,2),150,[1 0 0],'*')
end
Markers = ['o','d'];
for temp=1:2
        ydata_temp = Feat_Test(Label_Test==temp,:);
        scatter(ydata_temp(:,1),ydata_temp(:,2),150,Col(Label_Test==temp,:),Markers(temp),'filled')
end
set(gca,'fontsize',18)
% axis([ax1 ax2],[-2 3 -2 2])
%set(h,{'markers'},{12})
%set(h,'SizeData',96);
hold off
saveas(h,['Experiment',num2str(exp),'.png'])
