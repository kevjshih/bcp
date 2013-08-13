function D = pasc2D(testset, VOCopts)

%VOCyear =[];
%VOCopts = [];

cls = 'full';

if(~exist('VOCopts', 'var'))
   BDglobals;
   BDpascal_init;
end

VOCopts

if(exist('testset','var'))
   VOCopts.testset = testset;
end

ids = textread(sprintf(VOCopts.imgsetpath, VOCopts.testset), '%s');


t = tic;
for i = 1:length(ids)
   if(toc(t)>5)
      fprintf('%d/%d\n', i, length(ids));
      t = tic;
   end

   D(i).annotation.filename = [ids{i} '.jpg'];
   D(i).annotation.folder = VOCopts.dataset;
   objs = [];   
   try
      recs=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
   

      for o = 1:length(recs.objects)
         obj.polygon.pt = bbox2poly(recs.objects(o).bbox);
         obj.name = recs.objects(o).class;
         obj.partof = '';
         obj.partofobject = '';
         obj.id = o;

         objs = [objs obj];
      end
   end
   D(i).annotation.object = objs;
   D(i).annotation.train = 0;
   D(i).annotation.val = 0;
   D(i).annotation.test = 1;
end




function pts = bbox2poly(bbox);
% Grabbed from labelme
pts(1).x = num2str(bbox(1));
pts(1).y = num2str(bbox(2));

pts(2).x = num2str(bbox(1));
pts(2).y = num2str(bbox(4));

pts(3).x = num2str(bbox(3));
pts(3).y = num2str(bbox(4));

pts(4).x = num2str(bbox(3));
pts(4).y = num2str(bbox(2));

