function [cand2, cand] = getObjectCandidates(D, pred, imdir, maxcand, minprob, detections)
% cand = getObjectCandidates(D, detinfo, pred, thresh, imdir)

do_display = false;

ovthresh = 0.5;
maxc = 100;

for f = 1:numel(D)    
    
    if mod(f, 10)==0
        disp(num2str(f))
    end
    
    %minprob = 0.01;
    cand = getCandidateVotesWeighted(detections(f, :), pred, minprob);
       
    % Crop it
    info = imfinfo(fullfile(imdir, D(f).annotation.folder, D(f).annotation.filename));
    imh = info.Height;  imw = info.Width;
    cand.bbox_cropped = max(cand.bbox, 1);   
    cand.bbox_cropped = min(cand.bbox, repmat([imw imh imw imh], [size(cand.bbox, 1) 1]));
    
    % Cluster candidate detections
    [bbox2, w2, oldassign] = getCandidateClusters(cand.bbox_cropped, ovthresh, cand.w, 0.025, maxc);
    while 1                
        [bbox2, w2, assign] = getCandidateClusters(bbox2, ovthresh, w2, 0.01);
        if numel(assign)==numel(oldassign)
            break;
        end
        oldassign = assign;
    end
    
    %  
    [sv, si] = sort(w2, 'descend');
    bbox2 = bbox2(si, :);
    bbox2 = min(bbox2, repmat([imw imh imw imh], [size(bbox2, 1) 1]));
    [w2, ids, w_ids] = getBBoxCandidateSupport(bbox2, cand, ovthresh);
    
    n = min(maxcand, numel(w2));
    cand2(f).bbox = bbox2(1:n, :);
    cand2(f).w = w2(1:n);
    cand2(f).n = n;

   %im = imread(fullfile(imdir, D(f).annotation.folder, D(f).annotation.filename));
   %keyboard
end


return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{                      
    if do_display
        im = imread(fullfile(imdir, D(f).annotation.folder, D(f).annotation.filename));
        disp(num2str(sort(w2, 'descend')'))        
      
        [sv, si] = sort(w2, 'descend');   
               
        % get ground truth polygon
        gtbb = zeros(0, 4);
        for k = 1:numel(D(f).annotation.object)
            if isempty(D(f).annotation.object(k).partofobject)
                [x, y] = getLMpolygon(D(f).annotation.object(k).polygon);  
                gtbb(end+1, :) = [min(x) min(y) max(x) max(y)];
                
                ov = bbox_overlap_mex(gtbb(end, :), bbox2);                              

                disp([num2str(size(gtbb, 1)) ': ' D(f).annotation.object(k).name]) 
                i = min(find(ov(si)>0.5));            
                disp(['first: ' num2str([i ov(si(i)) sv(i)])]) 
                [mv, mi] = max(ov(si));
                disp(['best: ' num2str([mi mv sv(mi)])])

            end
        end        
        
        % show boxes        
        figure(1), hold off, imagesc(im), axis image, axis off, hold on
        nb = min(1, size(bbox2, 1));
        for k2 = nb:-1:1
            %col = [1-(k2-1)/max(nb-1,1) 0 (k2-1)/max(nb-1,1)];
            v = w2(k2) / max(w2);
            col = [v 0 1-v];            
            plot(bbox2(si(k2), [1 1 3 3 1])', bbox2(si(k2), [2 4 4 2 2])', 'Linewidth', 5, 'Color', col);
        end
        drawnow;
        
        [mv, mi] = max(w2);
        colors = 'gbycmkwgbycmkwg';
        [animalNames, animalPartNames, vehicleNames, vehiclePartNames, ...
             holdoutNames, animalScNames, vehicleScNames] = getDetectorNames;
         names = {};
         tsz=60;
        if ~isempty(ids)
            pnum = unique(cand.prednum(ids{mi}));
            n=0;
            for k2 = 1:numel(pnum);

                ind = find(cand.prednum(ids{mi})==pnum(k2)); 
                bbid = unique(cand.bbox_id(ids{mi}(ind)));
                bbox_src = cand.bbox_src{pnum(k2)}; %(bbid, :);
                
                ispart = strcmpAny([animalPartNames vehiclePartNames], detinfo(pnum(k2)).name);                                
                if (ispart && sum(w_ids{mi}(ind))>0.005)  || (~ispart && sum(w_ids{mi}(ind))>0.1)
                    n=n+1;
                    names{n} = detinfo(pnum(k2)).name;
                    names{n}(names{n}=='_') = ' ';
                    [mv2, mi2] = max(w_ids{mi}(ind));
                    %mi2 = find(w_ids{mi}(ind)>0.1*mv2);
                    mi2 = cand.bbox_id(unique(ids{mi}(ind(mi2))));
                    if isempty(mi2), keyboard; end
                    disp([names{n} ': ' num2str(mi2)])
                    hold on, plot(bbox_src(mi2, [1 1 3 3 1])', bbox_src(mi2, [2 4 4 2 2])', colors(n), 'LineWidth', 2);                    
                    disp([detinfo(pnum(k2)).name ': ' num2str(sum(w_ids{mi}(ind))) ' ' colors(n)])
                    text(10, size(im,2)/size(im,1)*tsz*n, names{n}, 'Color', colors(n), 'FontSize', 18)
                end                                                
                
            end
            
        end
        n=0;
        
        if 1
        figure(2), hold off, imagesc(im), axis image, axis off, hold on
        colors = 'gbycmrkwgbycmkwg';
        for k = 1:numel(cand.bbox_src)        
            ispart = strcmpAny([animalPartNames vehiclePartNames], detinfo(k).name);       
            shown = false;
            if ~isempty(cand.bbox_src{k}) %&& sum(cand.w(cand.prednum==1))>0.                
               
                for k2 = 1:size(cand.bbox_src{k},1)                                    
                    if sum(cand.w(cand.bbox_id==k2 & cand.prednum==k)) > (0.1*(~ispart) + 0.01*ispart)                        
                        if ~shown
                            n=n+1;
                            names{n} = detinfo(k).name;
                            names{n}(names{n}=='_') = ' ';                             
                            shown = true;                            
                        end                        
                        hold on, plot(cand.bbox_src{k}(k2, [1 1 3 3 1])', cand.bbox_src{k}(k2, [2 4 4 2 2])', colors(n), 'LineWidth', 2);                    
                    end
                end
                if shown
                    text(10, size(im,2)/size(im,1)*tsz*n, names{n}, 'Color', colors(n), 'FontSize', 18)
                end
            end                    
        end
        
        figure(3), hold off, imagesc(ones(size(im))), axis image, axis off, hold on
        hold on, plot([1 1 size(im,2) size(im,2) 1], [1 size(im,1) size(im,1) 1 1], 'k')
        n = 0;
        for k = 1:numel(cand.bbox_src)
            c = '';
            if strcmp(detinfo(k).name, 'dog')
                c = 'g';
                mi = 1;
                n=n+1;
                name = 'dog';
            elseif strcmp(detinfo(k).name, 'eye')
                c ='b';
                mi = 2;
                n=n+1;
                name = 'eye';
            elseif strcmp(detinfo(k).name, 'nose')
                c = 'r';
                mi = 2;
                n=n+1;
                name = 'nose';
            end
            if isempty(c), continue; end
                
%             boxw = [];
%             for k2 = 1:size(cand.bbox_src{k},1)
%                 boxw(k2) = sum(cand.w(cand.bbox_id==k2 & cand.prednum==k));
%             end            
%             [mv, mi] = max(boxw);
            ind = (cand.bbox_id==mi) & (cand.prednum==k);
            disp(sum(ind))
            ind = find(ind);
            if numel(ind)>10
                rp =randperm(numel(ind));
                ind = ind(rp(1:10));
            end
            hold on, plot(cand.bbox_src{k}(mi, [1 1 3 3 1])', cand.bbox_src{k}(mi, [2 4 4 2 2])', c, 'LineWidth', 4)
            hold on, plot(cand.bbox_cropped(ind, [1 1 3 3 1])'+rand(1)*0, cand.bbox_cropped(ind, [2 4 4 2 2])'+rand(1)*0, c, 'LineWidth', 1);           
            text(10, size(im,2)/size(im,1)*tsz*n, name, 'Color', c, 'FontSize', 18)
        end   
        end
        
    end
end
%}
% print -f1 -depsc ~/data/attributes/cvpr2010/results/vote_x_1.eps    
% print -f2 -depsc ~/data/attributes/cvpr2010/results/vote_x_2.eps
%print -f3 -depsc ~/data/attributes/cvpr2010/results/vote_dog_3.eps
