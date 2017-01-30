nc = 3; %numnber of clusters
N = 1000; % number of samples
D = 10; % Dimesions
X = rand(N,D)*100; % random points
C = datasample(X,nc,1)+1e-5; % pick random points as centroids; 1e-5 is added to avoid zero distance and Nan in U

%dist = Yf_EuDistArrayOfVectors1 (C, X); % fill the distance matrix
dist = zeros(nc, N);
for k = 1:nc,
	dist(k, :) = sqrt( sum(((X-ones(N, 1)*C(k, :)).^2)') ); %Distance between all datasamples and each centroid
end
% calculate new U, suppose m != 1
m = 1.2;
tmp = dist.^(-2/(m-1));
U = tmp./(ones(nc, 1)*sum(tmp));  % mu ie Memebership computed during training
[~,Idx_U] = max(U);   % index of centroid with more membership is assigned as index for sample


%%
D = pdist2(X,C);   % compute eucledian distance
[~,Idx_D] = min(D'); %index of minimum distance is assigned as index for sample

if Idx_U==Idx_D
    a = 'Pass'
else
    a = 'Both are different'
end
