set = 'train';
cls = 'aeroplane';
parts_inds = 2;

email = 'jiaa1@illinois.edu';
subject = ['Training \"' set '\" set ' cls ' parts'];

%% Retrieve the parts
all_parts = get_manual_parts(VOCopts, set, cls);
parts = {};
for parts_ind = parts_inds
   parts{end+1} = all_parts{parts_ind};
end

%% Select the applicability of the parts to the specified set
for parts_i = 1:length(parts)
   part = parts{parts_i};
   manual_set_applicability(VOCopts, part, set, false);
end

%% Initialize the part models
for parts_i = 1:length(parts)
   part = parts{parts_i};
   manual_refine_part(VOCopts, part, true);

   % Also train automatically, for comparison
   auto_refine_part(VOCopts, part);
end

%% Refine the parts
train_by_stage = false;
if train_by_stage
   for i = 1:2
      % Do stage 0 refinement
      for parts_i = 1:length(parts)
         part = parts{parts_i};
         manual_refine_part(VOCopts, part);
      end

      message = 'Ready for input';
      system(['echo ' message ' | mail -s "' subject '" ' email]);

      % Do stage 1 refinement (requires user input)
      for parts_i = 1:length(parts)
         part = parts{parts_i};
         manual_refine_part(VOCopts, part);
      end

      % Do stage 2 refinement
      for parts_i = 1:length(parts)
         part = parts{parts_i};
         manual_refine_part(VOCopts, part);
      end
   end
else
   for parts_i = 1:length(parts)
      part = parts{parts_i};

      done = false;
      while ~done
         % Do stage 0 refinement
         manual_refine_part(VOCopts, part);

         % Do stage 1 refinement (requires user input)
         message = 'Ready for input';
         system(['echo ' message ' | mail -s "' subject '" ' email]);
         manual_refine_part(VOCopts, part);
         done = ~strcmp('Yes', questdlg('Continue refining?'));
         
         % Do stage 2 refinement
         manual_refine_part(VOCopts, part);
      end
   end
end

message = 'Finished training';
system(['echo ' message ' | mail -s "' subject '" ' email]);
