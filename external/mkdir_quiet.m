function mkdir_quiet(dir)

if(~exist(dir, 'file'))
   mkdir(dir);
end
