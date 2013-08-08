function [cached_scores min_conf] = add_incremental_feat(cached_scores, varargin)%roc, num_bins)
% Split examples into num_bins bins based on where their score falls in the precision-recall curve

if(nargin==2 && isfield(varargin{1}, 'part')) % Just copy features of previous rounds
   model = varargin{1};
   
   computed = find([model.part.computed]);
   for i = 1:length(cached_scores)
      if(isempty(cached_scores{i}.labels))
         cached_scores{i}.incremental_feat = [];
      else
         cached_scores{i}.incremental_feat = [cached_scores{i}.region_score cached_scores{i}.part_scores]; 
         for j = computed(:)'
            offset = j + length(model.region_model);
            cached_scores{i}.incremental_feat(isinf(cached_scores{i}.incremental_feat(:, offset)), offset) = model.part(j).bias;
         end
      end
   end
else % 
   if(nargin==3)
      roc = varargin{1};
      num_bins = varargin{2};
   elseif(nargin==2)
      min_conf = varargin{1};
      num_bins = length(min_conf)-1;
   end
   
   if(~exist('min_conf','var'));
      % 1) Split pr curve
      prec_bins = linspace(0, 1, num_bins + 1);
      
      min_conf = zeros(1, num_bins+1);
      min_conf(1) = -inf;
      min_conf(num_bins+1) = inf;
      
      for i = 2:num_bins
         min_ind = max(find(roc.p>prec_bins(i)));
         min_conf(i) = roc.conf(min_ind);
      %   min_rec(i) = roc.r(min_ind);
      end
   end
   
   % 2) Bin the examples
   for i = 1:length(cached_scores)
      if(isempty(cached_scores{i}.labels))
         cached_scores{i}.incremental_feat = zeros(0, num_bins);
      else
         [dk bins] = histc(cached_scores{i}.scores, min_conf);
         N = numel(cached_scores{i}.labels);
         cached_scores{i}.incremental_feat = accumarray([(1:N)', bins], 1, [N, num_bins]);
      end
   end
end
