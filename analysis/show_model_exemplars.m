function show_model_exemplars(models)

if(isstruct(models))
    models = num2cell(models.part);
end

BDglobals;

% Extract 
shape = [3 3];
page = prod(shape);

for i = 1:length(models)
   fn = regexp(models{i}.name, '\d{4}_\d{6}', 'match');
   bb0 = regexp(models{i}.name, '\[\d+ \d+ \d+ \d+\]', 'match');
   bb = str2num(bb0{1}(2:end-1));

   im = imread(fullfile(im_dir, [fn{1} '.jpg']));
   
   figure(ceil(i/page));
   subplot(shape(1), shape(2),mod(i-1, page)+1);

   imagesc(im);
   axis off; axis image;
   hold on;
   draw_bbox(bb);
   if(isfield(models{i}, 'bb'))
        draw_bbox(models{i}.bb(1:4), 'r', 'linewidth', 3);
   end
   
    title(num2str(i));
    
end
