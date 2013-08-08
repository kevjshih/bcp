function feats = hyp_to_feat(model, hyp, feat_data)

if(isfield(model, 'loc_model') && model.loc_model==1)
  % Simple!
  feats = [cat(2, feat_data{[hyp.feat_ind]}); cat(2, hyp.score)];
else
  feat = feat_data.feat;
  padded = zeros(size(feat));
  
  padx = ceil(model.maxsize(2)/2+1);
  pady = ceil(model.maxsize(1)/2+1);
  
  feats = cell(length(hyp), 1);
  
  computed = [model.part.computed];
  
  % Create spatial filter for smoothed response, this should probably be paramaterized
  %f = [0.5; 1; 0.5]*[0.5 1 0.5];
  do_transform = isfield(model, 'do_transform') && model.do_transform==1;
  do_shift_rot = (isfield(model, 'shift') && numel(model.shift)>1) || (isfield(model, 'rotation') && numel(model.rotation)>1);
  
  for r = 1:length(hyp)
    feat_t = cell(model.num_parts, 1);
    
    for p = 1:model.num_parts
      if(computed(p))
        %feat_t{p} = hyp(r).score(p);
        % Skip it
      else
         loc = hyp(r).loc(p,:);
         x = loc(1);
         y = loc(2);
         s = loc(3);
         
         if(do_shift_rot)
            feat_ind = sub2ind(size(feat), s, loc(5), loc(6), max(loc(7),1)); 
         else
            feat_ind = s;
         end

         if(0 &&~padded(feat_ind)) % This way we only have to do it once ... % Padding is precomputed now
            padded(feat_ind) = 1;
            feat{feat_ind} = padarray(feat{feat_ind}, [pady padx 0], 0);
         end

         feat0 = feat{feat_ind}(y:y+model.part(p).size(1)-1, x:x+model.part(p).size(2)-1, :);

         if(do_transform)
            if(loc(4) == 2) % Flip it!
               feat0 = flipfeat(feat0);
            end
         end

         if(isfield(model.part(p), 'whiten'))
            feat_t{p} = model.part(p).whiten.W*(feat0(:) - model.part(p).whiten.mu(:));
         else
            feat_t{p} = feat0(:);
         end
      end
        
%        if(isfield(model, 'subset_split') && model.subset_split>0)
%          Nf = numel(feat_t{p});
%          Nsp = model.subset_split;
%          feat_t{p} = [zeros(Nf*(loc(8)-1),1); feat_t{p};
%          zeros(Nf*(Nsp-loc(8)), 1); accumarray(loc(8), 1, [Nsp, 1])];
    end
  
    % Score from previously cached detectors
    feat_score = hyp(r).cached_score(:);
    feats{r} = cat(1, feat_t{:}, feat_score); % Concatenate everything
  end
  
  % each column is a feature set for one hypothesis
  feats = cat(2, feats{:});

end
