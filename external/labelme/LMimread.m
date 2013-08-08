function img = LMimread(D, ndx, HOMEIMAGES)
%
% Reads one image from the database. 
%     img = LMimread(database, ndx, HOMEIMAGES);
%
% This function is similar to LMread. Just for compatibility.

filename = fullfile(HOMEIMAGES, D(ndx).annotation.folder, D(ndx).annotation.filename);
filename = strrep(filename, '\', '/');

if ~strcmp(filename(1:5), 'http:');
    filename = strrep(filename, '%20', ' '); % replace space character
end
%filename

img = imread(filename);


