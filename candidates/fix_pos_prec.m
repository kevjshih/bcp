function pos_prec = fix_pos_prec(pos_prec, D, cached_scores, cls)

added = 0;
for i = 1:length(D)
   boxes = LMobjectboundingbox(D(i).annotation, cls);
   
   if(~isempty(boxes) && ~any(cached_scores{i}.labels>0)) % This case was missed in original version
      for j = 1:length(pos_prec)
         pos_prec{j}{i} = zeros(1, size(boxes,1));
      end
      added = added + size(boxes,1);
   end
end

fprintf('Added %d new examples\n', added);
