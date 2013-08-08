function hash = make_hash(words, values)
% hash = make_hash(words)
% 
% Input:
%     words - an M element cell array of words (keys) to be stored in hash table
%             (Periods and spaces are replaced with underscore (_) to be compatible
%             with matlab structures)
%     values - an optional M element array of values to be associated with the corresponding key
%              if not provided, the index of the key will be used
%
% Output:
%     hash - hash table to be used with lookup_hash.m

words = strrep(words, ' ', '_');
words = strrep(words, '.', '_');

if(~exist('values', 'var'))
   values = 1:length(words);
end


hash_data(1:2:2*length(words)) = words;
hash_data(2:2:end+1) = mat2cell(values, 1, ones(1, length(words)));

hash = struct(hash_data{:});
