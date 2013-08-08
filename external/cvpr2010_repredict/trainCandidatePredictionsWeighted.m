function pred = trainCandidatePredictionsWeighted(detinfo, detections, dataset, thresh)
%
% detinfo(ndetectors).(name, datadir,A,B)
nmsthresh = 0.5;
ovthresh = 0.6; % Detected boxes should be 60% within GT
ndetectors = numel(detinfo);
pred = repmat(struct('name', '', 'offset', zeros(numel(dataset)*10,4), 'w', zeros(numel(dataset)*10,1), 'n', 0), ndetectors, 1);

for k = 1:numel(detinfo)
  pred(k).name = detinfo(k).name;
%  pred(k).all_names = detinfo(k).all_names;
end

for f = 1:numel(dataset) 
  if mod(f, 100)==0
      disp(num2str(f)); 
  end
  ann = dataset(f).annotation;  
  
  bboxes_d = cell(numel(detinfo), 1);
  
  for o = 1:numel(ann.object)
    
    object = ann.object(o);
    [x,y] = getLMpolygon(object.polygon); % object bounding box
    bbo = [min(x) min(y) max(x) max(y)];    
    ho = bbo(4)-bbo(2); wo = bbo(3)-bbo(1);  
    cxo = (bbo(1)+bbo(3))/2;  cyo = (bbo(2)+bbo(4))/2;
    
    if isempty(object.partof) || strcmp(object.partof,'0') % Corresponds to whole object (could be a part in CORE)
        objind = [];
        objind(1) = o;
        parts = strcmp({ann.object.partof}, num2str(o));
        objind = [o find(parts)];
        gtnames = {ann.object(objind).name};
        for k = 1:numel(objind),             
            gtname = ann.object(objind(k)).name;
            gtname(gtname==' ') = '_';   
            for di = 1:numel(detinfo)
                bboxes = detections{f, di};
                if isempty(bboxes) %~any(strcmp(gtname, detinfo(di).all_names)) || isempty(bboxes)
                    continue;
                end
                
                [x,y] = getLMpolygon(ann.object(objind(k)).polygon);
                gtbb = [min(x) min(y) max(x) max(y)]; % gt bounding box for detections (could be part)

                ov = bbox_contained(bboxes(:, [1:4]), gtbb([1:4])); %bbox_overlap_mex(gtbb([1:4]), bboxes(:, [1:4]));
                correct = ov>=ovthresh & bboxes(:, end)>=thresh;
                bbox = bboxes(correct, :); % correct detections
                nbb = size(bbox,1);

                % compute offsets
                centers = [mean(bbox(:, [1 3]),2) mean(bbox(:, [2 4]),2)];
                bbsize = [bbox(:, 3)-bbox(:,1)+1  bbox(:,4)-bbox(:,2)+1];            

                % ACCOUNT FOR FLIP HERE!!
                dx = cxo - centers(:, 1);
               
                flipped = bbox(:, 5)==2;
            
                dx(flipped) = centers(flipped, 1) - cxo; % I think this is right...

                 dy = cyo - centers(:, 2);
                dsx = wo ./ bbsize(:, 1);  dsy = ho ./ bbsize(:, 2);
                n = pred(di).n;
                pred(di).offset(n+(1:nbb), :) = [dx./bbsize(:,1) dy./bbsize(:,2) dsx dsy];
                pred(di).w(n+(1:nbb)) = bbox(:, end); % 1; % Not sure why it used equal weighting before...
                pred(di).n = n+nbb;            
            end
        end
    end
  end
end
  


for k = 1:numel(pred)
  pred(k).offset(pred(k).n+1:end, :) = [];
  pred(k).w(pred(k).n+1:end) = [];
  pred(k).w = pred(k).w/sum(pred(k).w);%ones(size(pred(k).w)) / pred(k).n;
end  
