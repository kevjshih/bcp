function all_names = getDetectorInfo(names)
% detinfo = getDetectorInfo(D, datadir)
%
% Gets information about the detectors whose output is stored in a given
% directory, including the detector name and the applicable subclasses.
if(~iscell(names))
    names = {names};
end

for k = 1:length(names)
    switch names{k}
        case 'animal'
            all_names{k} = {'bat', 'eagle', 'penguin', 'camel', 'dog', 'elephant', ...
                'elk', 'lizard', 'whale', 'monkey', 'crow', 'dolphin', 'cat', 'cow', 'alligator'};
        case 'bat,eagle' %flight_animal
            all_names{k} =  {'bat', 'eagle', 'crow'};
        case 'four_legged'
            %all_names{k} = {'camel', 'dog', 'elephant', ...
            %    'elk', 'lizard', 'monkey', 'cat', 'cow', 'alligator'};
            all_names{k} = {'camel', 'dog', 'elephant', ...
                'elk', 'cat', 'cow','horse', 'sheep'};
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
        otherwise
            all_names{k} = names{k};
    end
end

all_names = [all_names{:} names]; % Make sure to include original names as well
