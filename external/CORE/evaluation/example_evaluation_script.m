% example_evaluation_script

% datafile for annotations (these will change)
datafn = './release/data/CORE/annotations/dataset.mat';
imdir = './release/data/CORE/images/';

% -------- Do Not Change ---------- %
ann = [D.annotation];
testind = logical([ann.test]);
clear ann; 
areaThresh = 24*24; % this should stay fixed
ovThreshObj = 0.5;
ovThreshPart = 0.25;
% --------------------------------- %

warning off; % doing imfinfo on some images will create warnings

[animalNames, animalPartNames, vehicleNames, vehiclePartNames, holdoutNames] = getDetectorNames;

%% example of superordinate-category (vehicle) detection evaluation
detfn = '~/data/attributes/cvpr2010/candidatePredictorResultsVehicle.mat';
load(detfn, 'cand_vehicle');
bbox = {cand_vehicle(testind).bbox};
scores = {cand_vehicle(testind).w};
dncnames = {'vehicle'}; % this will ignore vehicles that are not fully annotated
result_vehicle = evaluateDetections(D(testind), imdir, vehicleNames, dncnames, holdoutNames, ...
      bbox, scores, ovThreshObj, areaThresh);
roc_vehicle = analyzeResult(result_vehicle);

%% example of part (head) detection evaluation 
name = 'head';
detdir = '~/data/attributes/cvpr2010/detections/';
for f = 1:numel(D)  
  if testind(f)
    ann = D(f).annotation;    
    fn = fullfile(detdir, ann.folder, [strtok(ann.filename, '.')  '_felz_det_' name]);        
    load(fn, 'bboxes');        
    keep = bboxNonMaxSuppression(bboxes(:, 1:4), bboxes(:, end), ovThreshPart);
    bbox{f} = bboxes(keep, 1:4);
    scores{f} = bboxes(keep, end);    
  end
end
bbox = bbox(testind);
scores = scores(testind);
dncnames = {'animal'}; % this will ignore parts within animals that are not fully annotated
result_head = evaluateDetections(D(testind), imdir, {name}, dncnames, holdoutNames, ...
      bbox, scores, ovThreshPart, areaThresh);
roc_head = analyzeResult(result_head);

%% example of basic category (dog) detection evaluation
name = 'dog';
detdir = '~/data/attributes/cvpr2010/detections/';
for f = 1:numel(D)
  if testind(f)
    ann = D(f).annotation;    
    fn = fullfile(detdir, ann.folder, [strtok(ann.filename, '.')  '_felz_det_' name]);        
    load(fn, 'bboxes');        
    keep = bboxNonMaxSuppression(bboxes(:, 1:4), bboxes(:, end), ovThreshPart);
    bbox{f} = bboxes(keep, 1:4);
    scores{f} = bboxes(keep, end);    
  end  
end
bbox = bbox(testind);
scores = scores(testind);
dncnames = {'animal'}; % this will ignore parts within animals that are not fully annotated
result_dog = evaluateDetections(D(testind), imdir, {name}, dncnames, holdoutNames, ...
      bbox, scores, ovThreshObj, areaThresh);
roc_dog = analyzeResult(result_dog);


