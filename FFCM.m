function mi = FFCM(Feat_Train,Label_Train)
addpath(genpath('fcmc'))
addpath(genpath('tsne'))
if size(Feat_Train,2)>2
%     ydata = tsne(Feat_Train, Label_Train);
else 
    ydata = Feat_Train;
end
nc = 2; m=1.1; max_iter=1000; term_thr=1e-15; eta = 2.0;
mi ={};
t=1;
k=max(Label_Train);%number of classes
Feat_Train_plt = Feat_Train;
Label_Train_plt = Label_Train;
Markers = ['o','d','+','*','x','s','^',...
    'v','>','<','p','h','.','o','d','+','*','x','s','^',...
    'v','>','<','p','h','.'];
if k<=5
    Cols = [0 0 1;1 0 0; 1 0 1; 1 1 0; 0 1 1  ];%0 1 0; 0 1 1 ;1 0 0; 1 0 1;1 1 0; 1 1 1
elseif k<=20
    Cols = [Cols;Cols*0.25;Cols*0.5;Cols*0.75];
else
Cols = rand(k,3);
end
temp_temp=0;
while (size(Feat_Train,1)>0) && (temp_temp<30)
    temp_temp = temp_temp+1;
    for temp = 1:k
        if size(Feat_Train(Label_Train==temp,:),1)>=nc
            kk = Feat_Train(Label_Train==temp,:);
            init_V = kk(1:nc, :);
            [mi_s,U,E] = Yf_FCMC1 (Feat_Train(Label_Train==temp,:),nc, [m; max_iter; term_thr;  1; 1], init_V);
%             w = Yf_PCMC1_FindWeights1(Feat_Train(Label_Train==temp,:),U,mi_s,m,1);
%             [mi_s,U,E] = Yf_PCMC1 (Feat_Train(Label_Train==temp,:),nc, w, [m; max_iter; term_thr;  1; 1], mi_s);
            
            
%             fuzzy probablistic c means
            [mi_s,U,E] = Yf_FPCMC1 (Feat_Train(Label_Train==temp,:),nc, [m;eta; max_iter; term_thr;  1; 1], init_V);
            
%             mi_s = fcm(Feat_Train(Label_Train==temp,:),nc, [m; max_iter; 1e-15;  1]);
%             [~,mi_s] = kmeans(Feat_Train(Label_Train==temp,:),nc);
            if temp_temp>1
                mi{temp} = [mi{temp};mi_s];
            else mi{temp} = mi_s;
            end
        else
            if temp_temp>1
                mi{temp} = [mi{temp};Feat_Train(Label_Train==temp,:)];
            else mi{temp} = Feat_Train(Label_Train==temp,:);
            end
        end
    end
    t=t+1;
    idx=[];
    Feat_Train=Feat_Train_plt;
    Label_Train=Label_Train_plt;
    m_clus={};
    for temp=1:k
        idx = [idx min(pdist2(Feat_Train,mi{temp}),[],2)];
    end
    [kk,Idx] = min(idx,[],2);
    corre = find(Idx==Label_Train);
    Feat_Train(corre,:)=[];
    Label_Train(corre)=[];
    idx2=[];
    for temp=1:k
        idx2 = [idx2 min(pdist2(Feat_Train_plt,mi{temp}),[],2)];
    end
    [kk,Idx2] = min(idx2,[],2);
    Col = Cols(Idx2,:);
    fig_plot=1;
    if fig_plot>0
    %close all
    h = figure,hold on
    for temp=1:k
        ydata_temp = ydata(Label_Train_plt==temp,:);
        scatter(ydata_temp(:,1),ydata_temp(:,2),[],Col(Label_Train_plt==temp,:),Markers(temp))
    end
    hold off
    saveas(h,[num2str(temp_temp),'_PeaksFile.jpg'])                                      
    end
end

