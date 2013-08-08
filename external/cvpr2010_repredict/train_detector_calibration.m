function params = train_detector_calibration(D, cached_scores, cls)


for i = 1:size(cached_scores{1}.part_scores,2)
   fprintf('Training part %d\n', i);
   [dk dk dk dk roc] = test_part_detections_D(cls, D, cached_scores, i);
   %figure(i);
   clf;
   params{i} = learn_obj_prob(roc, 2);
end
