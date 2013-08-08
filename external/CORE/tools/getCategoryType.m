function [isanimal, isvehicle, ispart, isblc, issc] = getCategoryType(name)
% [isanimal, isvehicle, ispart, isblc, issc] = getCategoryType(name)
% 
% Given a detector name, returns whether the name corresponds to an animal
% or vehicle, a part or basic category or superordinate category.


[animalNames, animalPartNames, vehicleNames, vehiclePartNames, ...
    holdoutNames,animalScNames, vehicleScNames] = getDetectorNames; 

isanimal = false;
isvehicle = false;
ispart = false;
isblc = false;
issc = false;

if any(strcmp(animalNames, name)) || any(strcmp(animalPartNames, name)) ...
        || any(strcmp(animalScNames, name))
    isanimal = true;
end
if any(strcmp(vehicleNames, name)) || any(strcmp(vehiclePartNames, name)) ...
        || any(strcmp(animalScNames, name))
    isvehicle = true;
end

if any(strcmp(vehiclePartNames, name)) || any(strcmp(animalPartNames, name))
    ispart = true;
end

if any(strcmp(vehicleNames, name)) || any(strcmp(animalNames, name))
    isblc = true;
end

if any(strcmp(vehicleScNames, name)) || any(strcmp(animalScNames, name))
    issc = true;
end
