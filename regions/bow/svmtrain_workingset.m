function  approxmodel = svmtrain_workingset(labels, features, opts, param)
% Speed things up when a small fraction of examples are support vectors

dofull = 1;

if(dofull)
   labels = labels(:);
   N = numel(labels);
    tic;
   model = svmtrain(labels, features, opts);
   approxmodel = compute_approx_model(model,param);
   toc;
else
   labels = labels(:);
   N = numel(labels);
   inds = rand(N,1)<(2000/N); % Start by sampling 2000 examples, don't know if there's a better way...
   inds0 = zeros(N,1);

   while sum(inds==1 & inds0==0)>10%0.001 % While any examples are added to working set
      tic;
         model = svmtrain(labels(inds), features(inds, :), opts);
      
      approxmodel = compute_approx_model(model,param);
      clear model;
      d = svmpredict_approx(features, approxmodel);

      inds0 = inds;
      fancy_cache = 1;
      if(fancy_cache)
         cand_inds = (d.*labels)<1.1; % Using 1.1 since predicted scores are approximate
         %rem_inds = inds0 & ~cand_inds;
         keep_inds = inds0 & cand_inds;
   
         new_inds = find(cand_inds & ~inds0);
   
         % Select top 2000 of the newest inds
         [a b] = sort(d(new_inds).*labels(new_inds), 'ascend'); % Pick the lowest scoring ones
   
         keep_inds(new_inds(b(1:min(2000,end)))) = 1;
   
         inds = keep_inds;
      else
         inds = (d.*labels)<1.1; % Using 1.1 since predicted scores are approximate
      end
      fprintf('Added %d examples, removed %d examples (%d in)\n', sum(inds==1 & inds0==0), sum(inds==0 & inds0==1), sum(inds==1));
      toc;
   end
end
