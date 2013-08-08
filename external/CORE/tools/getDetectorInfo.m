function detinfo = getDetectorInfo(D, datadir)
% detinfo = getDetectorInfo(D, datadir)
%
% Gets information about the detectors whose output is stored in a given
% directory, including the detector name and the applicable subclasses.

if ~exist('datadir', 'var') || isempty(datadir)
    datadir = '~dhoiem/data/attributes/cvpr2010/detections/';
end

fn = fullfile(datadir, D(1).annotation.folder, [strtok(D(1).annotation.filename, '.') '_felz_det*']);
files = dir(fn);
names = {files.name};
for k = 1:numel(names)
    names{k} = names{k}(numel([strtok(D(1).annotation.filename, '.') '_felz_det'])+2:end-4);
    all_names{k} = names{k};
    switch names{k}
        case 'animal'
            all_names{k} = {'bat', 'eagle', 'penguin', 'camel', 'dog', 'elephant', ...
                'elk', 'lizard', 'whale', 'monkey', 'crow', 'dolphin', 'cat', 'cow', 'alligator'};
        case 'bat,eagle' %flight_animal
            all_names{k} =  {'bat', 'eagle', 'crow'};
        case 'four_legged'
            all_names{k} = {'camel', 'dog', 'elephant', ...
                'elk', 'lizard', 'monkey', 'cat', 'cow', 'alligator'};
        case 'water_animal'
            all_names{k} = {'penguin', 'whale','dolphin','alligator'};
        case 'mammal'            
            all_names{k} = {'bat', 'camel', 'dog', 'elephant', ...
                'elk', 'whale', 'monkey', 'dolphin', 'cat', 'cow'};
        case 'vehicle_sc'
            all_names{k} = {'airplane', 'blimp', 'car', 'hovercraft', 'snowmobile', ...
                'semi', 'motorcycle', 'boat', 'ship', 'bus', 'carriage', 'bicycle', 'jetski'};
        case 'wheeled_vehicle'
            all_names{k} = {'car', 'semi', 'bus', 'carriage'};            
        case 'watercraft'            
            all_names{k} = {'hovercraft', 'boat', 'ship', 'jetski'};            
        case 'landcraft'
            all_names{k} = {'car', 'hovercraft', 'snowmobile', ...
                'semi', 'motorcycle', 'bus', 'carriage', 'bicycle'};                 
        case 'air_vehicle'
            all_names{k} = {'airplane', 'blimp'};                    
        case 'side_window,window'
            all_names{k} = {'side_window', 'window'};
        case 'windshield,front_windshield'
            all_names{k} = {'windshield', 'front_windshield'};            
        case 'row of windows'
            all_names{k} = {'row_of_windows'};                        
        case 'side mirror'
            all_names{k} = {'side_mirror'};                                    
    end
    
end

for k = 1:numel(names)
    detinfo(k).name = names{k};
    detinfo(k).all_names = all_names{k};
    detinfo(k).datadir = datadir;
end