function model = initialize_goalsize_model(im, bbox, init_params)
%% Initialize the exemplar (or scene) such that the representation
% which tries to choose a region which overlaps best with the given
% bbox and contains roughly init_params.goal_ncells cells, with a
% maximum dimension of init_params.MAXDIM
% Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('init_params','var')
  init_params.sbin = 8;
  init_params.hg_size = [8 8];
  init_params.MAXDIM = 10;
end

if ~isfield(init_params,'MAXDIM')
  init_params.MAXDIM = 10;
  fprintf(1,'Default MAXDIM is %d\n',init_params.MAXDIM);
end


[xs ys levels bboxes f_real scales] = exhaustive_init(im, bbox);


%todo = ceil(linspace(1, size(xs,1), 100));
todo = 1:size(xs,1);
for ind = 1:length(todo) %linspace(1, size(xs%1:size(xs,1)
   i = todo(ind);
   curfeats = f_real{levels(i)}(ys(i, 1):ys(i,2), xs(i,1):xs(i,2), :);
   
   model(ind) = model_given_feats(curfeats, init_params, im);
   model(ind).bb = bboxes(i, :);
end


function model = model_given_feats(curfeats, init_params, I)

model.init_params = init_params;
model.hg_size = size(curfeats);
model.mask = logical(ones(model.hg_size(1),model.hg_size(2)));

%fprintf(1,'hg_size = [%d %d]\n',model.hg_size(1),model.hg_size(2));
model.w = curfeats - mean(curfeats(:));
model.b = 0;
model.x = curfeats;

%Normalized-HOG initialization
model.w = reshape(model.x,size(model.w)) - mean(model.x(:));

%Fire inside self-image to get detection location
model.x = model.x(:);
%[model.bb, model.x] = get_target_bb(model, I);
model.bb = []; % I'll fill this in...

function [targetlvl, xind, yind] = get_best_feat(f, sc, box, sz)
% Find the best box for a given target size
[xs0 ys0] = meshgrid(1:size(f{1},2), 1:size(f{1},1));

best_ov = 0;

for i = 1:length(f)
    [Sy Sx dc] = size(f{i});
    xs = xs0(1:Sy, 1:Sx);
    ys = ys0(1:Sy, 1:Sx);
    
    bx = rootbox_simp(xs(:), ys(:), 8/sc(i), 0, 0, sz(1:2));
    
    [ovt indt] = max(bbox_overlap_mex(bx, box));
    
    if(ovt>best_ov)
        best_ov = ovt;
        targetlvl = i;
        xind = xs(indt);
        yind = ys(indt);
    end
end

fprintf('Best ov:%f\n', best_ov);

xind = [xind xind + sz(2)-1];
yind = [yind yind + sz(1)-1];


function [target_bb,target_x] = get_target_bb(model,I)
%Get the id of the top detection
mmm{1}.model = model;
mmm{1}.model.hg_size = size(model.w);
localizeparams.thresh = -100000.0;
localizeparams.TOPK = 1;
localizeparams.lpo = 10;
localizeparams.SAVE_SVS = 1;
localizeparams.FLIP_LR = 0;
localizeparams.pyramid_padder = 5;
localizeparams.dfun = 0;


[rs,t] = localizemeHOG(I,mmm,localizeparams);
target_bb = rs.bbs{1}(1,:);
target_x = rs.xs{1}{1};
