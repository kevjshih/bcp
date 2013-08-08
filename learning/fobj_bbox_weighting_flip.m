function [f g] = fobj_bbox_weighting(w, targets, data)

w = reshape(w, [], 4); 

f0 = zeros(length(targets), 4);


for i = 1:length(targets)
   B = data{i}(:, 1:4);
   probs = data{i}(:, 5);
   flipped = data{i}(:, 6);

   % For any part that has been flipped, the L/R weights get flipped
   flipped_weights = w;
   flipped_weights(flipped==1, [1 3]) = flipped_weights(flipped==1, [3 1]);

   Bp = bsxfun(@times, B, probs);
   
   num = sum(flipped_weights.*Bp, 1)';
   Z = flipped_weights'*probs+eps; % Make sure we don't divide by zero

   pred = num./Z;

   err = pred' - targets(i, :);
   f0(i, :) = (err).^2;

   if(any(isnan(f0(i,:))))
       keyboard
   end
   dg_h = bsxfun(@times, Bp, Z');
   dh_g = probs*num';
   g0{i} = bsxfun(@times, (dg_h - dh_g), 2*err./(Z'.^2));
   g0{i}(flipped==1, [1 3]) = g0{i}(flipped==1, [3 1]); % Flip things back...
end

if(toc>5)    
   imagesc(w);
   title(sprintf('%s', toc));
    tic;
   drawnow;
end


%f = f0;%
f = sum(f0(:));
g = sum(cat(3, g0{:}),3);
g = g(:);
