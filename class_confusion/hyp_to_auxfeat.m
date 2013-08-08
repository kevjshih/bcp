function feats = hyp_to_auxfeat(model, hyp, feat_data, aux_im, aux_ind, aux_max_c_pyramid_lvl)

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
        loc = hyp(r).loc(p,:); % loc contain the features options derived in inference
        x = loc(1);
        y = loc(2);
        s = loc(3);
        
        if(do_shift_rot)
          feat_ind = sub2ind(size(feat), s, loc(5), loc(6), max(loc(7),1));
        else
          feat_ind = s;
        end
        
        if(~padded(feat_ind)) % This way we only have to do it once ...
          padded(feat_ind) = 1;
          feat{feat_ind} = padarray(feat{feat_ind}, [pady padx 0], 0);
        end
        
        feat0 = feat{feat_ind}(y:y+model.part(p).size(1)-1, x:x+model.part(p).size(2)-1, :);
        
        if(do_transform)
          if(loc(4) == 2) % Flip it!
            feat0 = flipfeat(feat0);
          end
        end
        
        if(0&&isfield(model.part(p), 'spat_w') && ~isempty(model.part(p).spat_w))
          xbin = loc(end-2);
          ybin = loc(end-1);
          sbin = loc(end);
          
          spat_feats = zeros(size(model.part(p).spat_w));
          sc_feats = zeros(size(model.part(p).scal_w));
          if(ybin>size(spat_feats,1) || xbin>size(spat_feats,2) || ybin<1 || xbin<1)
            error('Spatial bins are out of bounds!\n');
          end
          spat_feats(ybin, xbin) = 1;
          %spat_feats = filter2(f, spat_feats); % Smooth it
          sc_feats(sbin) = 1;
          
          if(do_transform)
            if(loc(4) == 2) % Flip it!
              spat_feats = spat_feats(:, end:-1:1);
            end
          end
          
          feat_t{p} = [feat0(:); model.spat_weight*spat_feats(:); model.spat_weight*sc_feats(:)];
        else
          feat_t{p} = feat0(:);
        end
        
        if(isfield(model, 'subset_split') && model.subset_split>0)
          Nf = numel(feat_t{p});
          Nsp = model.subset_split;
          feat_t{p} = [zeros(Nf*(loc(8)-1),1); feat_t{p}; zeros(Nf*(Nsp-loc(8)), 1); accumarray(loc(8), 1, [Nsp, 1])];
        end
      end
    end
    % Score from previously cached detectors
    feat_score = hyp(r).cached_score(:);
    feats{r} = cat(1, feat_t{:}, feat_score); % Concatenate everything
  end
  
  % each column is a feature set for one hypothesis
  feats = cat(2, feats{:});

  % add color features
  if nargin >= 6 % if there are enough arguments, then add color features
    feat_color = add_feat_color(model, hyp, aux_im, aux_ind, aux_max_c_pyramid_lvl);
    feats = cat(1,feats,feat_color);
  end
end
