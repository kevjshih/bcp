function new_learner = boost_train(feat, labels, learner_type, ...
                                    columns, varargin)
% boost_train trains the boosted classifier that combines all of
% the part scores into a final object score

if(~exist('learner_type', 'var'))
   learner_type = 'thresh';
end

if(~exist('columns', 'var') || isempty(columns))
   columns = 1:size(feat,2);
end

% Remove localization errors
%feat(labels==0,:) = [];
%labels(labels==0) = [];

labels(labels==0) = -1; % Make sure to include localization errors

%feat(isinf(feat)) = -10000;

switch learner_type
    case 'svm'
      feat(:, end) = feat(:,end)/500;
      feat = feat(:, columns);
      feat_approach = 'a';
      switch feat_approach
         case 'a'
            feat = [feat isinf(feat)]';
         case 'b'
            feat = [feat mean(isinf(feat),2)]';
         case 'c'
            feat = [feat]';
      end


        feat(isinf(feat)) = 0;
        w = fast_svm(labels, feat, varargin{1});
        new_learner.model = w(:);
   case 'thresh'
      range(1) = quantile(feat(labels==-1), 0.5);
      range(2) = quantile(feat(labels==1), 0.95);

      thresholds = linspace(range(1), range(2), 10); % 10 evenly spaced thresholds
%      keyboard
      new_learner = real_boost_thresholded(feat, labels);
%      new_learner = real_boost_thresh_exhaust(feat, cached_scores, labels);
   case 'binned'
      new_learner.bins = [-1 -0.5 0 0.5];
      new_learner.weights = logit_boost(feat, cached_scores, labels, new_learner.bins);
   case 'dt'
      %feat = [feat isinf(feat)];
%      w = zeros(size(labels));
%      w(labels==1) = 1/(2*sum(labels==1));
%      w(labels==-1) = 1/(2*sum(labels==-1));
      %new_learner.model = train_boosted_dt_2c(feat, (size(feat,2)/2+1):size(feat,2), labels, 50, 2, 0);

      labels(labels==0) = -1;
      new_learner.model = train_boosted_dt_2c(feat(:, columns), [], labels, 50, 4, 0);
    case {'sigmoid_java', 'thresh_java', 'sigmoid_inf_java'}
         % Default params
         method = strrep(learner_type, '_java', '');

         numiter = 100;
         monotonic = 0;

         if(numel(varargin)>=1)
            monotonic = varargin{1};
         end

         if(numel(varargin)>=2)
            numiter = varargin{2};
         end

         fprintf('Training with %d iterations\n', numiter);
         feat(:,end) = -feat(:, end);

         if(exist('columns', 'var'))
            [data_train, learners] = process_data_java(feat, [], [], labels, columns, method, monotonic);
         else
            [data_train, learners] = process_data_java(feat, [], [], labels, [], method, monotonic);
         end   

         data_train = single(data_train);
   
         labels(labels<=0) = -1;
 
         weights = ones(1, numel(labels));
         weights(labels == 1) = 0.5*weights(labels == 1)./sum(labels == 1);
         weights(labels == -1) = 0.5*weights(labels == -1)./sum(labels == -1);
    
         new_learner.model = javaboost.boosting.LogitBoost.trainConcurrent(data_train, labels, learners, numiter, 8, weights);
   case 'sigmoid_java_inf' % Include bias when scores are equal to -inf
        feat(:,end) = -feat(:, end);
      if(exist('columns', 'var'))
         [data_train, learners] = process_data_java(feat, [], [], labels, columns, 'sigmoid_inf');
      else
         [data_train, learners] = process_data_java(feat, [], [], labels, [], 'sigmoid_inf');
      end   
       data_train = single(data_train);
   
      labels(labels<=0) = -1;
 
       weights = ones(1, numel(labels));
       weights(labels == 1) = 0.5*weights(labels == 1)./sum(labels == 1);
       weights(labels == -1) = 0.5*weights(labels == -1)./sum(labels == -1);
    
       new_learner.model = ...
           javaboost.boosting.LogitBoost.trainConcurrent(data_train, labels, learners, 100, 8, weights);
   case 'sigmoid_java_co'
     disp('using sigmoid w co');
     method = strrep(learner_type, '_java', '');

     numiter = 70;
     monotonic = 0;
     
     if(numel(varargin)>=1)
         monotonic = varargin{1};
     end
     
     if(numel(varargin)>=2)
         numiter = varargin{2};
     end
     
     fprintf('Training with %d iterations\n', numiter);
     feat(:,end) = -feat(:, end);
     
     if(exist('columns', 'var'))
         [data_train, learners] = process_data_java(feat, [], [], labels, columns, method, monotonic);
     else
         [data_train, learners] = process_data_java(feat, [], [], labels, [], method, monotonic);
     end   

     

     for i = 1:7
         for j = i+1:15
             learners.add(javaboost.weaklearning ...
                          .MultiFeatureLRLearner([i-1, j-1] ...
                                                         ));
         end
     end


     data_train = single(data_train);
     
     labels(labels<=0) = -1;
     
     weights = ones(1, numel(labels));
     weights(labels == 1) = 0.5*weights(labels == 1)./sum(labels == 1);
     weights(labels == -1) = 0.5*weights(labels == -1)./sum(labels == -1);
     
     new_learner.model = ...
         javaboost.boosting.LogitBoost.trainConcurrent(data_train, labels, learners, 100, 10, weights);     
end

new_learner.type = learner_type;
new_learner.columns = columns;
