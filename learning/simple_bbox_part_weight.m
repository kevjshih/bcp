function w = simple_bbox_part_weight(targets, data)

prob_sum = zeros(size(data{1}, 1), 1);
err_sum = zeros(size(data{1}, 1), 4);

% For each part, collect the box prediction error
for i = 1:length(targets)
   B = data{i}(:, 1:4);
   probs = data{i}(:, 5);
   flipped = data{i}(:, 6);

   % For any part that has been flipped, the L/R weights get flipped
   err = (B - targets(i, :)).^2;
   err_flip = err;
   err_flip(flipped==1, [1 3]) = err(flipped==1, [3 1]);

   prob_sum = prob_sum + probs;
   err_sum = err_sum + err_flip;
end


average_err = (1./prob_sum')*err_sum;
