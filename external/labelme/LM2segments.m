function [img, seg, names, counts] = LM2segments(D, imagesize, HOMEIMAGES, HOMELMSEGMENTS)
%
% Transforms all the labelme labels into segmentation masks.
% It takes into account the occlusions, removing labeled pixels that are
% occluded by other objects. Depth ordering is achieved using the function
% LMsortlayers.m
%
%  [img, seg, names, counts] = LM2segments(D, imagesize, HOMEIMAGES,
%  HOMELMSEGMENTS)
% 'img' and 'seg' are matrices
%
% To precompute the segmentations:
%
%   LM2segments(D, [], HOMEIMAGES, HOMELMSEGMENTS) % removing the second
%     argument, makes the segmentation mask to have the same size than the
%     images.
%
% To read the precomputed segmentations:
%   [img, seg, names] = LM2segments(Dt(1), [], HOMEIMAGES, HOMELMSEGMENTS);
%
%
% HOMEANNOTATIONS = 'http://labelme.csail.mit.edu/Annotations'
% HOMEIMAGES = 'http://labelme.csail.mit.edu/Images'

if nargin==4
    precomputed = 1;
    
    if length(D) == 1
        fileseg = fullfile(HOMELMSEGMENTS, D(1).annotation.folder, [D(1).annotation.filename(1:end-4) '.mat']);
        % read the image and return
        if exist(fileseg, 'file')
            load(fileseg)
            seg = S;
            img = [];
            return
        end
    end
else
    precomputed = 0;
    HOMELMSEGMENTS = '';
end

Nimages = length(D);

% Create list of objects:
[names, counts, imagendx, objectndx, objclass_ndx] = LMobjectnames(D, 'name');
for k = 1:length(objclass_ndx)
    D(imagendx(k)).annotation.object(objectndx(k)).namendx = objclass_ndx(k);
end

if nargout>0
    if Nimages > 1
        % Initalize output variables
        seg = zeros([imagesize(1) imagesize(2) Nimages], 'uint16');
        img = zeros([imagesize(1) imagesize(2) 3 Nimages], 'uint8');
    end
end

if Nimages > 1
    figure
end

for ndx = 1:Nimages
    % Load image
    imgtmp = LMimread(D, ndx, HOMEIMAGES);
    annotation = D(ndx).annotation;
    if size(imgtmp,3)==1; imgtmp = repmat(imgtmp, [1 1 3]); end
    [nrows ncols cc] = size(imgtmp);
    
    % Scale image so that image box fits tight in image
    if ~isempty(imagesize)
        scaling = max(imagesize(1)/nrows, imagesize(2)/ncols);
        [annotationtmp, imgtmp] = LMimscale(annotation, imgtmp, scaling, 'bicubic');

        % Crop image to final size
        [nr nc cc] = size(imgtmp);
        sr = floor((nr-imagesize(1))/2);
        sc = floor((nc-imagesize(2))/2);
        [annotationtmp, imgtmp] = LMimcrop(annotationtmp, imgtmp, [sc+1 sc+imagesize(2) sr+1 sr+imagesize(1)]);

        Mclasses = zeros([imagesize(1) imagesize(2)]);
    else
        annotationtmp = annotation;

        Mclasses = zeros([size(imgtmp,1) size(imgtmp,2)]);
    end
    
    if isfield(annotationtmp, 'object')
        % Sort layers
        annotationtmp = LMsortlayers(annotationtmp, imgtmp);

        % Get segmentation
        [mask, classes] = LMobjectmask(annotationtmp, size(imgtmp));
        classesndx = [annotationtmp.object.namendx];
        area = squeeze(sum(sum(mask,1),2));

        j = find(area>2); % remove small objects
        mask = mask(:,:,j);
        classesndx = classesndx(j);

        % Assign labels taking into account occlusions!
        for k = size(mask,3):-1:1;
            Mclasses = Mclasses+classesndx(k)*(Mclasses==0).*mask(:,:,k);
        end
    end

    if nargout>0
        % Store values
        seg(:,:,ndx)   = uint16(Mclasses);
        img(:,:,:,ndx) = imgtmp;
    end

    % Save gist if a HOMELMSEGMENTS file is provided
    if precomputed
        I = imgtmp; 
        S = uint16(Mclasses);
        mkdir(fullfile(HOMELMSEGMENTS, D(ndx).annotation.folder))
        fileseg = fullfile(HOMELMSEGMENTS, D(ndx).annotation.folder, [D(ndx).annotation.filename(1:end-4) '.mat']);
        if ~isempty(imagesize)
            save (fileseg, 'I', 'S', 'names')
        else
            save (fileseg, 'S', 'names')
        end
    end

    % Visualization
    if Nimages > 1
        subplot(121)
        image(imgtmp); axis('equal'); axis('tight');
        title(sprintf('%d (out of %d)', ndx, Nimages))
        subplot(122)
        image(mod(Mclasses,256)); axis('equal'); axis('tight');
        colormap(gray(256))
        drawnow
    end
end

if Nimages > 1
    if nargout > 0
        % plot stats
        figure
        subplot(121)
        loglog(sort(counts, 'descend'))
        xlabel('count rank')
        ylabel('Number of instances')
        axis('tight')
        pixelcounts = hist(single(seg(:)), 0:single(max(seg(:))));
        unlabeled = pixelcounts(1);
        pixelcounts = pixelcounts(2:end); % remove unlabeled pixels;
        subplot(122)
        loglog(sort(pixelcounts, 'descend'))
        xlabel('area rank')
        ylabel('Number of pixels')
        axis('tight')
        title(sprintf('%d categories', length(names)))

        Ns = min(20, size(seg,3));
        figure
        montage(reshape(uint8(mod(seg(:,:,1:Ns),256)), [imagesize(1) imagesize(2) 1 Ns]))
    else
        figure
        loglog(sort(counts, 'descend'))
        xlabel('count rank')
        ylabel('Number of instances')
        axis('tight')
    end
end


