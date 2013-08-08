% trainCandidateVotingScript

thresh = 0.05; % minimum probability of object to be considered

% Assumes all data is in cached_scores

calib_param = train_detector_calibration(D, cached_scores, cls); % Done
cached_scores_calib = apply_detector_calibration(cached_scores, calib_param); % Done

detections = convert_detections(cached_scores_calib); % Done

%detector_info = model.part;

for i = 1:model.num_parts
   detector_info(i).all_names = num2str(i);
   detector_info(i).name = num2str(i);
end

% Select 
[Dpos inds] = LMquery(D, 'object.name', cls, 'exact');
pred = trainCandidatePredictionsWeighted(detector_info, detections(inds, :), Dpos, thresh); % Just need to account for flip!
pred = compressCandidatePredictor(pred, 0.85);


% Now test the model!
%imdir = '~/prog/attributes/CORE/images_resized';
[cands2 cand] = getObjectCandidates(D, pred, im_dir, 200, 0.01, detections);

for i = 1:length(cands2)
   boxes{i} = [cands2(i).bbox cands2(i).w];
end

res = evaluateAUC('four_legged', boxes, D_test);
roc = analyzeResult(res);


return







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DO_ANIMAL = 0;
DO_ANIMAL_SC = 0;
DO_ANIMAL_BLC = 0;
DO_VEHICLE = 0;
DO_VEHICLE_SC = 1;
DO_VEHICLE_BLC = 0;

[animalNames, animalPartNames, vehicleNames, vehiclePartNames] = getDetectorNames;

%% Animals Full
if DO_ANIMAL
ind = false(size(detinfo));
for k = 1:numel(detinfo)
    if strcmpAny(detinfo(k).all_names, animalNames) || strcmpAny(detinfo(k).all_names, animalPartNames)
        ind(k) =true;
    end
end
pred_animal = trainCandidatePredictionsWeighted(detinfo(ind), D(trainind), thresh);
pred_animal = compressCandidatePredictor(pred_animal, 0.85);
save ~/data/attributes/cvpr2010/candidatePredictors_animal.mat pred_animal
end

%% Animals SC
if DO_ANIMAL_SC
ind = false(size(detinfo));
for k = 1:numel(detinfo)
    if strcmpAny(detinfo(k).all_names, animalNames) && ~strcmpAny(detinfo(k).all_names, animalPartNames)
        ind(k) =true;
    end
end
pred_animal_sc = trainCandidatePredictionsWeighted(detinfo(ind), D(trainind), thresh);
pred_animal_sc = compressCandidatePredictor(pred_animal_sc, 0.85);
save ~/data/attributes/cvpr2010/candidatePredictors_animal_sc.mat pred_animal_sc
end

%% Animals BLC
if DO_ANIMAL_BLC
ind = false(size(detinfo));
for k = 1:numel(detinfo)
    if strcmpAny(detinfo(k).name, animalNames) 
        ind(k) =true;
    end
end
pred_animal_blc = trainCandidatePredictionsWeighted(detinfo(ind), D(trainind), thresh);
pred_animal_blc = compressCandidatePredictor(pred_animal_blc, 0.85);
save ~/data/attributes/cvpr2010/candidatePredictors_animal_blc.mat pred_animal_blc
end

%% Vehicles Full
if DO_VEHICLE
ind = false(size(detinfo));
for k = 1:numel(detinfo)
    if strcmpAny(detinfo(k).all_names, vehicleNames) || strcmpAny(detinfo(k).all_names, vehiclePartNames)
        ind(k) =true;
    end
end
pred_vehicle = trainCandidatePredictionsWeighted(detinfo(ind), D(trainind), thresh);
pred_vehicle = compressCandidatePredictor(pred_vehicle, 0.85);
save ~/data/attributes/cvpr2010/candidatePredictors_vehicle.mat pred_vehicle
end


%% Vehicles SC
if DO_VEHICLE_SC
ind = false(size(detinfo));
for k = 1:numel(detinfo)
    if strcmpAny(detinfo(k).all_names, vehicleNames) && ~strcmpAny(detinfo(k).all_names, vehiclePartNames)
        ind(k) =true;
    end
end
pred_vehicle_sc = trainCandidatePredictionsWeighted(detinfo(ind), D(trainind), thresh); 
pred_vehicle_sc = compressCandidatePredictor(pred_vehicle_sc, 0.85);
save ~/data/attributes/cvpr2010/candidatePredictors_vehicle_sc.mat pred_vehicle_sc
end

%% Vehicles BLC
if DO_VEHICLE_BLC
ind = false(size(detinfo));
for k = 1:numel(detinfo)
    if strcmpAny(detinfo(k).name, vehicleNames) 
        ind(k) =true;
    end
end
pred_vehicle_blc = trainCandidatePredictionsWeighted(detinfo(ind), D(trainind), thresh);
pred_vehicle_blc = compressCandidatePredictor(pred_vehicle_blc, 0.85);
save ~/data/attributes/cvpr2010/candidatePredictors_vehicle_blc.mat pred_vehicle_blc
end
