clear
Class=[2,10,17,24];
%https://in.mathworks.com/help/stats/_bq9uxn4.html
load('list.mat')
data = {};

%%
load(list{2});
data{1} = [X Y];

%% 
load(list{10});
Y = zeros(size(species,1),1);
for Idx = 1: size(species,1)
    if strcmp(species{Idx},'setosa')
        Y(Idx)=1;
    end
    if strcmp(species{Idx},'versicolor')
        Y(Idx)=2;
    end
    if strcmp(species{Idx},'virginica')
        Y(Idx)=3;
    end
end
data{2} = [meas Y];
%%
load(list{17});
species = Y;
Y = zeros(size(species,1),1);
for Idx = 1: size(species,1)
    if strcmp(species{Idx},'g')
        Y(Idx)=1;
    end
    if strcmp(species{Idx},'b')
        Y(Idx)=2;
    end
end
data{3} = [X Y];


%%
load(list{24});
Y = zeros(size(grp,1),1);
species = grp;
for Idx = 1: size(species,1)
    if strcmp(species{Idx},'Cancer')
        Y(Idx)=1;
    end
    if strcmp(species{Idx},'Normal')
        Y(Idx)=2;
    end
end
data{4} = [obs Y];
save('Matlab_data.mat','data')