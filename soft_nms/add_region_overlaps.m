function cached_scores = add_region_overlaps(cached_scores, th, fgmr)

if(~exist('th', 'var'))
   th = 0.5;
end

if(~exist('fgmr', 'var'))
   fgmr = 1;
end

for i = 1:length(cached_scores)
   fprintf('%d/%d\n', i, length(cached_scores));
   if(isempty(cached_scores{i}.labels))
      cached_scores{i}.overlaps = sparse([]);
   else
      if(fgmr)
         ov = bbox_overlap_fgmr_mex(cached_scores{i}.regions, cached_scores{i}.regions);
%         ov(ov<th) = ;
         ov = max(min(ov,1), 0.5) - 0.5;
      else
         ov = bbox_overlap_mex(cached_scores{i}.regions, cached_scores{i}.regions);
         ov(ov<th) = 0;
      end
      ov = sparse(ov);
      cached_scores{i}.overlaps = ov;
   end
end
