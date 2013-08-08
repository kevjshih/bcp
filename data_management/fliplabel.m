function new_label = fliplabel(label)

    new_label = label;

      if(strmatch('L ', label))
         new_label(1:2) = 'R ';
      elseif(strmatch('R ', label))
         new_label(1:2) = 'L ';
      elseif(strmatch('Left ', label))
         new_label = strrep(label, 'Left ', 'Right ');
      elseif(strmatch('Right ', label))
         new_label = strrep(label, 'Right ', 'Left ');
      end