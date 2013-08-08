function new_dup = find_duplicates(existing, new_ex)

% Compute 
vals0 = sum(existing.^2, 1);
valsN = sum(new_ex.^2, 1);

new_dup = zeros(size(new_ex,2),1);

for i = 1:length(vals0)
   tocheck = abs(vals0(i)-valsN)<1e-9;
   
   if(any(tocheck))
      same = abs(vals0(i) - existing(:,i)'*new_ex(:, tocheck))<1e-9;

      %new_dup(tocheck) = new_dup(tocheck) | same;
      new_dup(tocheck) = reshape(new_dup(tocheck), [], 1) | same(:);
   end
end

