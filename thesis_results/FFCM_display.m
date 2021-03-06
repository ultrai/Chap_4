function mi = FFCM_display(Feat_Train,Label_Train)
addpath(genpath('fcmc')) %adding the FCM toolbox
addpath(genpath('tsne')) %adding the TSNE toolbox 
if size(Feat_Train,2)>2  %Condition to employ tsne
    Feat_Train_mean = mean(Feat_Train);
    Feat_Traint = Feat_Train-ones(size(Feat_Train,1),1)*Feat_Train_mean;
    Feat_Train_std = std(Feat_Traint);
    Feat_Traint = Feat_Train./(ones(size(Feat_Traint,1),1)*Feat_Train_std+0.01);
    ydata = tsne(Feat_Traint, Label_Train);
else 
    ydata = Feat_Train;
end
nc = 3; m=1.1; max_iter=100; term_thr=1e-15; eta = 2.0;  %standar values for FCm
mi ={};   %intialization of means set
t=1;
k=max(Label_Train);    %number of classes
Feat_Train_plt = Feat_Train;
Label_Train_plt = Label_Train;
Markers = ['o','d','+'];
Cols = [0 0 1;1 0 0; 0 1 0];
temp_temp=0; %counter for maximum # of iterations
while (size(Feat_Train,1)>0) && (temp_temp<30) %30
    temp_temp = temp_temp+1;
    for temp = 1:k %for each class
        if size(Feat_Train(Label_Train==temp,:),1)>=nc %# of samples should be greater to number of clusters
            kk = Feat_Train(Label_Train==temp,:); %selecting samples belonging to a class
            init_V = kk(1:nc, :);    % first two samples as cluster means 
            [mi_s,U,E] = Yf_FCMC1 (Feat_Train(Label_Train==temp,:),nc, [m; max_iter; term_thr;  1; 1], init_V);
%             w = Yf_PCMC1_FindWeights1(Feat_Train(Label_Train==temp,:),U,mi_s,m,1);
%             [mi_s,U,E] = Yf_PCMC1 (Feat_Train(Label_Train==temp,:),nc, w, [m; max_iter; term_thr;  1; 1], mi_s);
            
%             fuzzy probablistic c means
%             [mi_s,U,E] = Yf_FPCMC1 (Feat_Train(Label_Train==temp,:),nc, [m;eta; max_iter; term_thr;  1; 1], init_V);
            
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
%     for temp = 1:k %for each class
%         for m_idx =size(mi{temp},1):-1:1
%             [~,effec] = min(pdist2(Feat_Train,mi{temp}(m_idx,:)),[],1);
%             if Label_Train(effec)~=temp
%                 mi{Label_Train(effec)} = [mi{Label_Train(effec)};mi{temp}(m_idx,:)];
%                 mi{temp}(m_idx,:)=[];
%             end
%                 
%         end
%     end
    
    t=t+1;
    idx=[];
    Feat_Train=Feat_Train_plt;
    Label_Train=Label_Train_plt;
    m_clus={};
    for temp=1:k
        idx = [idx min(pdist2(Feat_Train,mi{temp}),[],2)];
%         [~,effec] = min(pdist2(Feat_Train,mi{temp}),[],2);
%         for m_idx = size(mi{temp},1):-1:1
%             if sum(effec==m_idx)==0
%                 mi{temp}(m_idx)=[];
%             end
%         end
    end
    [kk,Idx] = min(idx,[],2); % predicting the indexes
    corre = find(Idx==Label_Train); %Identifying correct predictions 
    Feat_Train(corre,:)=[];   %removing correc predictions 
    Label_Train(corre)=[];
    idx2=[];
    for temp=1:k
        idx2 = [idx2 min(pdist2(Feat_Train_plt,mi{temp}),[],2)];
    end
    [kk,Idx2] = min(idx2,[],2);
    Col = Cols(Idx2,:);
    fig_plot=1;
    close all
    h = figure,hold on
    ydata_temp = ydata(Label_Train_plt==1,:);
    scatter(ydata_temp(:,1),ydata_temp(:,2),[],Col(Label_Train_plt==1,:),Markers(1))
    ydata_temp = ydata(Label_Train_plt==2,:);
    scatter(ydata_temp(:,1),ydata_temp(:,2),45,Col(Label_Train_plt==2,:),Markers(2))
    scatter(mi{1}(:,1),mi{1}(:,2),45,'b','s')
    scatter(mi{2}(:,1),mi{2}(:,2),45,'r','*')
    if k==3
        ydata_temp = ydata(Label_Train_plt==3,:);
        scatter(ydata_temp(:,1),ydata_temp(:,2),45,Col(Label_Train_plt==3,:),Markers(3))
        scatter(mi{3}(:,1),mi{3}(:,2),'X')
    end
    %title(['Centroids 1(' num2str(size(mi{1},1)) ')   Centroids 2(' num2str(size(mi{2},1)),')' ])
    title(['Iteration ' num2str(temp_temp)]) 
    hold off
    set(gca,'fontsize',18)

    saveas(h,[num2str(temp_temp),'_PeaksFile.png'])                                      
end
temp_temp
end

