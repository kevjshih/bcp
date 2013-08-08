function cached_scores = apply_detector_calibration(cached_scores, calib_param)


for i = 1:length(cached_scores)
   if(~isempty(cached_scores{i}.regions))
      for j = 1:length(calib_param)
         cached_scores{i}.part_scores(:, j) = sigmoid(cached_scores{i}.part_scores(:, j), calib_param{j});
      end
   end
end
