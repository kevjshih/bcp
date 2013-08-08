function [w_noloo w_loo Cout ap] = train_gen_loo(labels, feats, Cs, imind)


if(length(Cs)==1)
   bestCind = 1;
else
   best_ap = -inf;

   r = randperm(max(imind));
   r = r(1:ceil(end/2));
   te = ismember(imind, r);
   
   feat_te = feats(:, te==1);
   feats_tr = feats(:, ~te);
   
   labels_te = labels(te==1);
   labels_tr = labels(~te);
   %imind = imind(~te);

   for i = 1:length(Cs)
      C = Cs(i);
      [w0 alpha0] = svm_dual_mex(labels_tr, feats_tr, C);


   % Compute LOO estimates
%   w_loo = compute_loo(labels, feats, C, imind, alpha0);
%   scores = classify_loo(feats, w0, w_loo, imind);

%   roc(i) = computeROC(scores, labels);
%   ap(i) = VOCap(roc(i).r(:), roc(i).p(:));

      scores_te = w0(1:end-1)*feat_te;
      roc_te(i) = computeROC(scores_te(:), labels_te);
      auc_true(i) = roc_te(i).area;
   %ap_true(i) = VOCap(roc_te(i).r(:), roc_te(i).p(:));
   end
%if(0)
%   if(ap(i)>best_ap) 
%      best_ap = ap(i);
%      w_noloo_out = w0;
%      w_loo_out = w_loo;
%      Cout = C;
%   end
%end

   [bestAUC bestCind] = max(auc_true);
   fprintf('**************Best AP: %f (%d:%f)\n', bestAUC, bestCind, Cs(bestCind));
end

[w_noloo alpha0] = svm_dual_mex(labels, feats, Cs(bestCind));
w_loo = compute_loo(labels, feats, Cs(bestCind), imind, alpha0);

function [w_loo] = compute_loo(labels, feats, C, imind, alpha0)
% Collect which images need loo estimate
im_todo = unique(imind(alpha0(:)>1e-6 & labels(:)==1));

w_loo = cell(max(imind), 1);

for i = im_todo(:)'
   touse = imind~=i;
   w_loo{i} = svm_dual_mex(labels(touse), feats(:, touse), C, [], alpha0(touse));
end

