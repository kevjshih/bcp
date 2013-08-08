[a host] = system('uname -n');

if(isempty(strfind(host, 'taub')))
   basedir = '/home/engr/iendres2/prog/boosted_detection/';%fileparts(which('load_init_data.m'))
   DATADIR = '/shared/daf/';
else   
   basedir = '/home/iendres2/prog/boosted_detection/';%fileparts(which('load_init_data.m'));
   DATADIR = '/projects/VisionLanguage/common';
end
