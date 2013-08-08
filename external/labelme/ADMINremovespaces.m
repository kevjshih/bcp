function ADMINremovespaces(HOMEANNOTATIONS, HOMEIMAGES, annotationfolder)
%
% Removes spaces, %2025 and %20 from filenames and replaces them with '_'
% This function is to be used with caution as it might affect other users.
% 
% Those characteres can be introduced sometimes when reading images from
% the web.
%
% ADMINremovespaces(HOMEANNOTATIONS, HOMEIMAGES, annotationfolder)

% Rename images

if length(HOMEIMAGES)>0
    files = dir(fullfile(HOMEIMAGES, annotationfolder, '*.jpg'));

    Nfiles = length(files);
    for i = 1:Nfiles
        filename = fullfile(HOMEIMAGES, annotationfolder, files(i).name);

        % rename image file
        src = filename;
        dest = removecaracters(files(i).name);
        cmd = sprintf('rename "%s" %s', src, dest)
        system(cmd)
    end
end

% Rename annotations and replace %2025 and %20 and spaces by '_' from annotation.filename
files = dir(fullfile(HOMEANNOTATIONS, annotationfolder, '*.xml'));

Nfiles = length(files);
for i = 1:Nfiles
    filename = fullfile(HOMEANNOTATIONS, annotationfolder, files(i).name);
    
    % rename image file indexed inside the file
    [fid, message] = fopen(filename,'r');
    if fid == -1; error(message); end
    xml = fread(fid, 'uint8=>char');
    fclose(fid);
    xml = xml';
        
    ii = strfind(xml, '<filename>');
    jj = strfind(xml, '</filename>');

    if length(i)>0
        tmp = removecaracters(xml(ii:jj));
        xml = [xml(1:ii-1) tmp xml(jj+1:end)];

        % save annotation
        fid = fopen(filename,'w');
        fprintf(fid, xml');
        fclose(fid);


        % rename annotation file
        src = filename;
        dest = removecaracters(files(i).name);
        cmd = sprintf('rename "%s" %s', src, dest)
        system(cmd)
    else
        disp(filename)
    end
end


function out = removecaracters(in)

out = strrep(in, ' ', '_');
out = strrep(out, '%2520', '_');
out = strrep(out, '%20', '_');

            
            

