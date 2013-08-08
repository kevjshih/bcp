function [annotation, img] = LMimflip(annotation, img);
%
% Crops an image and modifies the corresponding annotation.
%
% [annotation, img] = LMimcrop(annotation, img, [xmin xmax ymin ymax]);
%

[nrows ncols c] = size(img);

img = img(:, end:-1:1,:);

% Flip the polygon coordinates
if isfield(annotation, 'object')
    Nobjects = length(annotation.object); n=0;
    for i = 1:Nobjects
        Npoints = length(annotation.object(i).polygon.pt);
        clear X Y
        for j = 1:Npoints
            % Scale each point:
            x=str2num(annotation.object(i).polygon.pt(j).x);
%            y=str2num(annotation.object(i).polygon.pt(j).y);

            X(j) = round(ncols - x + 1); % crop(1) = 1 implies no crop
%            Y(j) = round(y - crop(3) +1);

            annotation.object(i).polygon.pt(j).x = num2str(X(j));
%            annotation.object(i).polygon.pt(j).y = num2str(Y(j));
        end
        % If the object is outside the image, mark as deleted
    end
end

