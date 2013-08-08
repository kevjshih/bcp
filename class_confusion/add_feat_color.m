function feat_color = add_feat_color(model, hyp, color_im, part_ind, pyramid_max_lvl)

if(~exist('pyramid_max_lvl','var'))
  pyramid_max_lvl = 4; % default
end
feat_color = cell(1,length(hyp));

% calculate expected number of color features (per color), so to pad all hyp 
% to have the same number of color features
expected_numfeat = 0;
numfeat_at_layer = 1;
for layer = 0:pyramid_max_lvl
  expected_numfeat = expected_numfeat + numfeat_at_layer;
  numfeat_at_layer = numfeat_at_layer * 4;
end

% convert RGB to YCbCr
color_im = rgb2ycbcr(color_im);
% rescale the image, so that svm behaves better (why though?)
color_im = im2double(color_im); 
% turn color image into a integral image
color_im = cumsum(cumsum(color_im,1),2);

% for each hypothesis, add the avg intensity of each lvl to feat_color
for h = 1:length(hyp)
  % fprintf('calculating color features of %d hypothesis\n',h);
  for color = 1:3
    im = color_im(:,:,color);
    
    bbox = hyp(h).bbox(part_ind,:);
    % get the biggest bounding box possible while being within bound of im
    upperleft = max(floor(bbox(1:2)),[1 1]);
    lowerright = min(ceil(bbox(3:4)), [size(im,2) size(im,1)]);
    % NOTE: the first coordinate (row) of im matrix corresponds to y,
    % while the second coordinate (column) of im matrix corresponds to x !!!
    % thus size(im) == [ max_y max_x ] ... (who made this decision?)
    im_bbox = im( upperleft(2):lowerright(2) , upperleft(1):lowerright(1) );
    
    feat_color{h} = cat(1,feat_color{h},feat_color_recursive(im_bbox,pyramid_max_lvl));
    % the recursive function divides the pyramid a total of max_lvl (usually 4) times

    if length(feat_color{h}) < expected_numfeat * color
      num_padding = expected_numfeat*color - length(feat_color{h});
      feat_color{h} = cat(1,feat_color{h},zeros(num_padding,1));
    end
  end
  
end
feat_color = cat(2,feat_color{:});

end

function feat_c = feat_color_recursive(im, lvl)

if(lvl == 0 || any(size(im)<=[2 2]) )
  feat_c = [average_intensity(im)];
  return;
end

feat_c = average_intensity(im);

ymid = floor(size(im,1)/2);
xmid = floor(size(im,2)/2);
feat_c = cat(1,feat_c, feat_color_recursive( im( 1:ymid,1:xmid ), lvl-1));
feat_c = cat(1,feat_c, feat_color_recursive( im( 1:ymid,xmid:end ), lvl-1));
feat_c = cat(1,feat_c, feat_color_recursive( im( ymid:end,1:xmid ), lvl-1));
feat_c = cat(1,feat_c, feat_color_recursive( im( ymid:end,xmid:end ), lvl-1));

end

% note that this is not really the average of the whole image,
% but the average of the im(2:end,2:end)
function avg_intensity = average_intensity(im)
total_intensity = im(end,end)+im(1,1)-im(end,1)-im(1,end);
num_pixels = (size(im,1)-1)*(size(im,2)-1);
if num_pixels == 0
  avg_intensity = 0.0;
else
  avg_intensity = total_intensity/num_pixels;
end

end

    
