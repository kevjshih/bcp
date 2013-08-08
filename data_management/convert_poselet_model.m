function model = convert_poselet_model(model0)

% Convert the poselet model to look as much like our models as possible
if(isfield(model0, 'model'))
    model0 = model0.model;
end

model.interval = 10;
model.sbin = 8;
model.fixed = 0;
model.maxsize = [0 0];
model.num_parts = str2num(model0.Attributes.num_poselets);
model.part = [];
model.cls = model0.Attributes.category;
model.is_poselet = 1;
model.thresh = -inf;
model.cached_weight = 0;

for i = 1:model.num_parts
    pl = model0.poselets.poselet{i};
    w0 = str2num(pl.svm_weights.Text);
    model.part(i).bias = w0(end);
    
    sz0 = str2num(pl.Attributes.dims);
    model.part(i).size = [sz0/8 - 1 36];
    model.part(i).filter = reshape(w0(1:end-1), model.part(i).size);
    model.part(i).computed = 0;
    model.part(i).name = '';
    model.part(i).spat_const = [];
    model.part(i).poselet_ind = i;
    model.maxsize = max(model.maxsize, model.part(i).size(1:2));
end

% Now sort things for more efficient detection
sizes = cat(1, model.part.size);
[a b inds] = unique(sizes, 'rows');
[dk resort] = sort(inds);
model.part = model.part(resort);
