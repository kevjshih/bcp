function [hyp feat_data] = inference(input, model, varargin)
% Input
%   a) YxXx3 image
%   b) input.feat
%      input.scales
% model ...
% regions ...
% Cached Data (for each region)
%  cached_scores
if(isfield(model, 'symbols')) % DPM model
   [hyp feat_data] = inference_region_dpm(input, model, varargin{:});
elseif(isfield(model, 'is_poselet') && model.is_poselet==1)
    tic;
   [hyp feat_data] = inference_poselet(input, model, varargin{:});
   toc;
elseif(isfield(model, 'subset_split') && model.subset_split>0)
   [hyp feat_data] = inference_split(input, model, varargin{:});
elseif(isfield(model, 'do_collect') && model.do_collect==1)
   [hyp feat_data] = inference_collect(input, model, varargin{:});
elseif(isfield(model, 'do_transform') && model.do_transform==1)
   if(~isfield(model, 'Ntodo') || model.Ntodo==1)
      [hyp feat_data] = inference_trans(input, model, varargin{:});
   else
      [hyp feat_data] = inference_trans_mult(input, model, varargin{:});
   end
elseif(isfield(model, 'loc_model') && model.loc_model==1)
   [hyp feat_data] = inference_loc(input, model, varargin{:});
else
   [hyp feat_data] = inference_simp(input, model, varargin{:});
end
