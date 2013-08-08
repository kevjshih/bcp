function result = evaluateDetections(D, imdir, gtnames, dncnames, holdoutNames, ...
  bbox_all, scores_all, ovthresh, areaThresh)
% result = evaluateDetections(D, imdir, gtnames, dcnames, holdoutNames, bbox_all, scores_all, ovthresh, areaThresh)
%
% Determines which of the input bounding boxes are true positives, which
% are false positives, and which are ignorable "do not cares".  Also
% includes additional information for later analysis.
%
% Input
%   D(nimages): the annotation structure for each image
%   imdir: the directory containing the images
%   gtnames: the names of ground-truth objects or parts that should be
%            considered correct
%   dcnames: the names of ground-truth objects that should be treated as
%            don't cares (e.g., "animal")
%   holdoutNames: the names of holdout categories (for later analysis)
%   bbox_all{nimages}: the detection bounding boxes in each image;
%                      bbox_all{f}(k, 1:4) = [x1 y1 x2 y2] for kth object
%   scores_all{nimages}: the score of each bounding box
%   ovthresh: the bounding box intersection over union threshold required
%             to be considered a correct detection
%             e.g., ovthresh = 0.5 for objects, 0.25 for parts
%   areaThresh: the minimum pixel area required to consider a ground truth
%               object as detectable (otherwise, it is "do not care")
%               e.g., areaThresh = 24*24;
%
% Output
%  result(nimages): stores bounding boxes, ground truth labels, and
%                   auxilliary information for later analysis


DISPLAY = false; % sets whether to print out progress update

% By default, ignoreExtraPos=false, so that multiple detections on the
% same object counts against you.  
if ~exist('ignoreExtraPos', 'var') || isempty(ignoreExtraPos)
    ignoreExtraPos = false;
end


%% Get result information for each image

result = []; 
for f = 1:numel(D)
    
    if DISPLAY
      if mod(f, 500)==0 % for progress update
          disp(num2str(f));
      end
    end
    
    %% Initialization
    ndnc = 0;
    
    ann = D(f).annotation;

    scores = scores_all{f};
    bbox = bbox_all{f};
    
    result(f).npos = 0;
    result(f).nholdout = 0;
    result(f).ndifficult = 0;        
    
    iminfo = imfinfo(fullfile(imdir,  ann.folder, ann.filename));
    imsize = [iminfo.Height iminfo.Width];  
    
    result(f).ov = zeros(numel(scores), 1, 'single');
    result(f).objnum = zeros(numel(scores), 1, 'uint16');
    labels = -ones(size(scores,1), 1, 'single');
    missing = [];
    nearest = [];
    isholdout_all = false(size(scores,1),1);
    
    if ~isempty(bbox)
                        
        % set bounds to lie within image
        bbox = [max(min(bbox(:, 1), imsize(2)), 1)  max(min(bbox(:, 2), imsize(1)), 1) ...
            max(min(bbox(:, 3), imsize(2)), 1)  max(min(bbox(:, 4), imsize(1)), 1)];    
    end
    
    %% Set ground truth values
    % For each object, get overlapping boxes and mark highest confidence
    % one as positive.
    if(~isfield(ann, 'object')) % Needed for incomplete V2
        ann.object = [];
    end
    for k = 1:numel(ann.object)

      gtname = ann.object(k).name;  % name of part or object
      gtname(gtname==' ') = '_';    % replace space with underscore

      objnum = k;

      thresh = ovthresh; 
      if ~isempty(ann.object(k).partof) 
          objnum = str2num(ann.object(k).partof);          
      end      
      objname = ann.object(objnum).name;
      if any(strcmp(objname, holdoutNames))
          isholdout = true;
      else
          isholdout = false;
      end

      % if the current object in ground truth structure is one of the given
      % target objects, then check whether it is detected
      if any(strcmp(gtname, gtnames{2}))
        % Find object
        parent = ann.object(str2double(ann.object(k).partof));
        if(~any(isacategory(gtnames{1}, parent.name)))
          continue; % Wrong object!
        end
      
        [x,y] = getLMpolygon(ann.object(k).polygon);
        gtbox = [min(x) min(y) max(x) max(y)];        

        area = (gtbox(3)-gtbox(1))*(gtbox(4)-gtbox(2));           

        isdifficult = area < areaThresh; % these are do not cares
        if ~isdifficult
            result(f).npos = result(f).npos+1;
            if isholdout
                result(f).nholdout = result(f).nholdout+1;
            end
        else
            result(f).ndifficult = result(f).ndifficult+1;
        end

        if ~isempty(bbox)

            ov = getBoxOverlap(gtbox([1 3 2 4]), bbox(:, [1 3 2 4]));
            ind = ov >= thresh;
            ind2 = ov > result(f).ov;             

            if ~any(ind) && ~isdifficult % missed detection
              missing = [missing ; k];
              [maxval, maxind] = max(ov);
              nearest(end+1) = maxind;                                
            else % object is detected          
              result(f).ov(ind2) = ov(ind2); % = max(ov, result(f).ov);
              result(f).objnum(ind2) = k;                

              if ignoreExtraPos 
                labels(labels==-1 & ind) = 0;
              end

              % ignore bounding boxes that are already considered to have
              % detected another object
              ind = ind & labels < 1; 
              tmpind = find(ind);

              if isempty(tmpind) && ~isdifficult  % missed detection
                missing = [missing ; k];
                nearest(end+1) = 0;
              else % it's detected
                [maxscore, maxind] = max(scores(tmpind, 1));            
                if isdifficult
                  labels(tmpind(maxind)) = 0;
                else
                  if isholdout                
                      isholdout_all(tmpind(maxind)) = 1;
                  end
                  labels(tmpind(maxind)) = 1;
                end
              end          
            end
        end
      end % end if target object
    end % end loop over objects
      
    %% Handle "don't cares"
    % If any object detection has 0.5 overlap with a don't care polygon, it
    % is treated as a "don't care".  If any part detection has a center
    % within a don't care polygon, it is treated as a "don't care".
    if ~isempty(bbox) 
        
        isdnc = false(size(labels));
      
        % check whether we are detecting parts or not (ispart)
        [isanimal, isvehicle, ispart, isblc, issc] = getCategoryType(gtnames{1});
        
        dnc_poly = [];
        for k = 1:numel(ann.object)
          if any(strcmp(ann.object(k).name, dncnames))
            [dnc_poly(end+1).x, dnc_poly(end+1).y] = getLMpolygon(ann.object(k).polygon);
          end
        end
        
        % for whole objects, bounding box must have 0.5 overlap to be a dnc
        dnc_bbox = zeros(numel(dnc_poly), 4);
        for k = 1:numel(dnc_poly)
            dnc_bbox(k, :) = [min(dnc_poly(k).x) min(dnc_poly(k).y) max(dnc_poly(k).x) max(dnc_poly(k).y)];   
        end        
        if isblc || issc            
            for k1 = 1:size(dnc_bbox, 1)
                isdnc = isdnc | reshape((getBoxOverlap(bbox(:,[1 3 2 4]), dnc_bbox(k1, [1 3 2 4]))>0.5),[],1);
            end
            
            %keyboard
        % for parts, center of bbox must be within poly to be a dnc    
        elseif ispart
            cx = (bbox(:, 1)+bbox(:,3))/2;
            cy = (bbox(:, 2)+bbox(:,4))/2;
            for k1 = 1:numel(dnc_poly)              
                isdnc = isdnc | inpolygon(cx, cy, dnc_poly(k1).x, dnc_poly(k1).y);
            end
        else
            error(['unknown type: ' gtnames{1}]);
        end
        
        ndnc = ndnc + sum(isdnc & labels==-1);
        labels(isdnc & labels==-1) = 0;
    end

    result(f).ndontcare = ndnc;
    result(f).isholdout = isholdout_all;
    result(f).scores = scores;
    result(f).bbox = bbox;
    result(f).labels = labels; 
    result(f).missing = missing;
    result(f).nearest = nearest;
    result(f).nfound = sum(result(f).labels==1);
    result(f).nmissing = numel(result(f).missing);  
end  
    
    
