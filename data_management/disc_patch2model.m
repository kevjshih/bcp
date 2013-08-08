function  model = disc_patch2model(model0, cls)

model = init_model(cls);

for i = 1:size(model0.w,1)
    part.w = reshape(model0.w(i,:), [8 8 31]);
    part.size = [8 8 31];
    part.b = model0.rho(i);
    
    model = add_model(model, part);
end