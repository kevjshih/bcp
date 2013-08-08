function [codebook idx pc_data] = create_sift_codebook(files, K, Npcs)

if(~exist('K','var'))
   K = 100;
end

for i = ceil(linspace(1,length(files), min(500,length(files))))
   fprintf('%d/%d\n', i, length(files));
   im = imread(files{i});

   data_all(i,:) = dense_sift(im, ceil(rand*5)+10);
end

data = cat(2, data_all{:,1});
clear data_all

%[tau, mu, sigma] = EM_init_kmeans(data_samp', K);
%[tau, mu, sigma] = EM(data_samp', tau, mu, sigma); %Priors, Mu, Sigma);


if(nargin>=3) % Compute and project with principle components
    means = mean(data, 2);
    [pc score latent] = princomp(data');
    pcs = pc(:,1:Npcs);
    
    pc_data.pcs = pcs;
    pc_data.means = means;
    pc_data.latent = latent(1:Npcs);
    
    data_samp = pca_project(data, pc_data);
   
    r = randperm(size(data_samp,2));
    data_samp = data_samp(:, r(1:min(100000,end)));
    
    [codebook idx energy] = vl_kmeans(data_samp, K, 'Algorithm', 'Elkan','numrepetitions', 3);%, 'distance','l1');
    
else
    r = randperm(size(data,2));
    data_samp = data(:, r(1:min(100000,end)));
    [codebook idx energy] = vl_kmeans(data_samp, K, 'Algorithm', 'Elkan','numrepetitions', 3);%, 'distance','l1');
end

codebook = double(codebook);
