function cached_scores = prune_viewpoint(D, cached_scores, cls, viewpoint)
% This should really be a D based operation
BDglobals;
BDpascal_init;

[dk pos_inds] = LMquery(D, 'object.name', cls, 'exact');

for i = pos_inds(:)'
   [dk bn] = fileparts(D(i).annotation.filename);

   recs=PASreadrecord(sprintf(VOCopts.annopath, bn));

   % Find any objects of class cls that have appropriate viewpoint
   obj = recs.objects;
   ok = strcmp({obj.view}, viewpoint) & strcmp({obj.class}, cls);
   % Keep regions that match
   ok_bboxes = cat(1, obj(ok).bbox);
   if(any(ok)) 
      ok_regions = max(bbox_overlap_mex(cached_scores{i}.regions, ok_bboxes),[],2)>=0.5;
   else
      ok_regions = false(size(cached_scores{i}.regions,1), 1);
   end

   cached_scores{i} = prune_cached_scores(cached_scores{i}, ok_regions);
end
