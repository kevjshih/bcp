function [parts] = manually_select_parts_fancy(ims, obj_bboxes, curr_parts, select_features_mask, grid_size)
% Select multiple parts with the ability to see selections and remove them.
%
% Input:
%   ims: cell array of full filenames of the images
%   obj_bboxes: cell array of bounding boxes of the objects in the
%               corresponding index of ims
%   curr_parts (optional): the parts from a previous round of selection.
%   select_features_mask (optional): flag for having the user draw a
%                                    feature mask
%   grid_size (optional): [height width] of the number of tiles to show
%                         per page of objects
%
% Output:
%   parts: the user-selected parts
%          cell array of structs containing the fields...
%            - im: full filename of the part's image
%            - obj_bbox: bounding boxes of the objects in im
%            - bbox: bounding box of the selected part
%            - features_mask: a mask of the user-drawn part; only set if
%                             select_features_mask is true
%            - icon: small image representative of the part, for previewing
%            - icon_bbox: bounding box of the selected part in the icon's
%                         coordinate system
%

parts = {};
if ~exist('curr_parts', 'var')
    curr_parts = {};
end

%% Initialize display settings.
ims_length = length(ims);
scrsz = get(0,'ScreenSize');
window_pos = [scrsz(3)/4 scrsz(4)/4 scrsz(3)/2 scrsz(4)/2];

if ~exist('select_features_mask', 'var')
    select_features_mask = false;
end

if ~exist('grid_size', 'var')
    grid_size = [5 7];
end

parts_per_page = grid_size(1);
parts_curr_page = 1;

object_ims_per_page = grid_size(1)*grid_size(2);
object_num_pages = ceil(ims_length/object_ims_per_page);
if exist('object_start_page', 'var')
    object_curr_page = max(1, min(object_num_pages, object_start_page));
else
    object_curr_page = 1;
end
crop_padding = 5;  % the amount of pixels to show around a crop

%% Create window and show loading screen.
window = figure('WindowKeyPressFcn', @windowKeyHandler, 'Position', window_pos, 'WindowStyle', 'modal');
loading_text = uicontrol('Parent', window, 'Style', 'text', 'FontSize', 32, 'Position', [0 0 window_pos(3) window_pos(4)]);

%% Preload the images.
I = cell(ims_length, 1);
for ims_i = 1:ims_length
    % Update the loading screen with the image loading progress.
    l = sprintf('Loading images... (%i/%i)', ims_i, ims_length);
    set(window, 'Name', l);
    set(loading_text, 'String', l);
    drawnow;
    
    % Load the image if it's a path and crop to the object box.
    I{ims_i} = convert_to_I(ims{ims_i});
    I{ims_i} = crop(I{ims_i}, obj_bboxes{ims_i}, crop_padding);
    
    % If the window was closed, quit.
    if ~ishandle(window)
        return;
    end
end

%% Load the display.
set(window, 'Name', 'Loading display...');
set(loading_text, 'String', 'Loading display...');
drawnow;

% Set up main display.
main_hbox = uiextras.HBox('Parent', window);

% Set up chosen curr_parts display.
parts_sidebar = uiextras.VBox('Parent', main_hbox, 'Padding', 5, 'BackgroundColor', [.4 .4 .4]);
% uicontrol('Parent', parts_sidebar, 'Style', 'text', 'String', 'Chosen curr_parts', 'FontSize', 10);
parts_vbox = uiextras.VBox('Parent', parts_sidebar, 'Padding', 5, 'Spacing', 5, 'BackgroundColor', get(parts_sidebar, 'BackgroundColor'));
parts_tiles = ones(grid_size(1), 1);
parts_axes = ones(grid_size(1), 1);
removeButtonHandlers = cell(parts_per_page, 1);
parts_remove_buttons = ones(parts_per_page, 1);
for i = 1:parts_per_page
    removeButtonHandlers{i} = createremoveButtonHandler(i);
    
    part_tile = uiextras.VBox('Parent', parts_vbox, 'BackgroundColor', [0 0 0]);
    parts_tiles(i) = part_tile;
    parts_axes(i) = axes('Parent', parts_tiles(i), 'ActivePositionProperty', 'Position');
    parts_remove_buttons(i) = uicontrol('Parent', parts_tiles(i), 'String', 'Remove', 'Callback', removeButtonHandlers{i});
    
    set(part_tile, 'Sizes', [-1 20]);
end
% Set up chosen curr_parts navigation
parts_nav_vbox = uiextras.VBox('Parent', parts_sidebar, 'Padding', get(parts_vbox, 'Padding'), 'BackgroundColor', get(parts_vbox, 'BackgroundColor'));
parts_nav_text = uicontrol('Parent', parts_nav_vbox, 'Style', 'text', 'String', '1/1', 'FontSize', 16, 'BackgroundColor', [1 1 1]);
parts_nav_buttons = uiextras.HBox('Parent', parts_nav_vbox);
parts_prev_button = uicontrol('Parent', parts_nav_buttons, 'Style', 'pushbutton', 'Callback', @partsPrevButtonHandler, 'String', '/\');
parts_next_button = uicontrol('Parent', parts_nav_buttons, 'Style', 'pushbutton', 'Callback', @partsNextButtonHandler, 'String', '\/');

% Set up finish button. Add an HBox for padding.
finish_hbox = uiextras.HBox('Parent', parts_sidebar, 'Padding', 5, 'BackgroundColor', get(parts_sidebar, 'BackgroundColor'));
uicontrol('Parent', finish_hbox, 'Style', 'pushbutton', 'Callback', @finishButtonHandler, 'String', 'Finish (f)', 'BackgroundColor', [.7 1 .7]);

set(parts_sidebar, 'Sizes', [-1 50 30]);

% Set up object choosing layout.
obj_hbox = uiextras.HBox('Parent', main_hbox);

object_vbox = uiextras.VBox('Parent', obj_hbox, 'Padding', 5, 'BackgroundColor', [.7 .7 .7]);
object_grid = uiextras.Grid('Parent', object_vbox, 'Padding', 5, 'Spacing', 5, 'BackgroundColor', get(object_vbox, 'BackgroundColor'));
object_nav_hbox = uiextras.HBox('Parent', object_vbox, 'Padding', get(object_grid, 'Spacing'), 'BackgroundColor', get(object_grid, 'BackgroundColor'));

% Set up object grid display.
% Create each tile (image axes + "selection" button) for the page display grid.
selectionButtonHandlers = cell(object_ims_per_page, 1);
object_tiles = ones(object_ims_per_page, 1);
object_axes = ones(object_ims_per_page, 1);
object_selection_buttons = ones(object_ims_per_page, 1);
for i = 1:object_ims_per_page
    selectionButtonHandlers{i} = createSelectionButtonHandler(i);
    
    object_tile = uiextras.VBox('Parent', object_grid);
    object_tiles(i) = object_tile;
    object_axes(i) = axes('Parent', object_tile, 'ActivePositionProperty', 'Position', 'ButtonDownFcn', selectionButtonHandlers{i});
    object_selection_buttons(i) = uicontrol('Parent', object_tile, 'String', 'Select', 'Callback', selectionButtonHandlers{i});
    
    set(object_tile, 'Sizes', [-1 20]);
end
set(object_grid, 'RowSizes', -ones(grid_size(1), 1), 'ColumnSizes', -ones(grid_size(2), 1));

% Set up navigation buttons.
object_prev_button = uicontrol('Parent', object_nav_hbox, 'Style', 'pushbutton', 'Callback', @objectPrevButtonHandler, 'String', '<');
object_nav_text = uicontrol('Parent', object_nav_hbox, 'Style', 'text', 'FontSize', 18, 'BackgroundColor', [1 1 1]);
object_next_button = uicontrol('Parent', object_nav_hbox, 'Style', 'pushbutton', 'Callback', @objectNextButtonHandler, 'String', '>');

set(object_vbox, 'Sizes', [-1 30]);

set(main_hbox, 'Sizes', [-1.5 -10]);

%% Clear the loading screen and show the display.
delete(loading_text);
set(window, 'Name', 'Select an object...');
redraw_object_display();
redraw_parts_display();

%% Let the user select images until the window is closed.
waitfor(window);

%% Set up helper functions.
    function windowKeyHandler(~, e)
        % Callback for keyboard events in the main window.
        
        % Check if the GUI is fully loaded (some keyboard shortcuts should
        % be ignored if the GUI isn't ready.
        gui_ready = ~ishandle(loading_text);
        
        switch e.Key
            case {'n', 'rightarrow'}
                if gui_ready
                    objectNextButtonHandler([], []);
                end
            case {'p', 'leftarrow'}
                if gui_ready
                    objectPrevButtonHandler([], []);
                end
            case {'u', 'uparrow'}
                if gui_ready
                    partsPrevButtonHandler([], []);
                end
            case {'d', 'downarrow'}
                if gui_ready
                    partsNextButtonHandler([], []);
                end
            case 'escape'
                close(window);
            case 'f'
                finishButtonHandler();
        end
    end

    function redraw_parts_display()
        for i = 1:parts_per_page
            parts_i = ((parts_curr_page-1)*parts_per_page)+i;
            
            curr_axes = parts_axes(i);
            curr_remove_button = parts_remove_buttons(i);
            
            if parts_i > length(curr_parts)
                % Hide tiles if there aren't enough pictures to fill them.
                set(parts_tiles(i), 'BackgroundColor', get(parts_vbox, 'BackgroundColor'));
                cla(curr_axes);
                set(curr_axes, 'Visible', 'off');
                set(curr_remove_button, 'Visible', 'off');
            else
                part = curr_parts{parts_i};
                
                % Show tile in case it was hidden before.
                set(parts_tiles(i), 'BackgroundColor', [0 0 0], 'ButtonDownFcn', removeButtonHandlers{i});
                set(curr_remove_button, 'Visible', 'on');
                
                % Show icon.
                hold(curr_axes, 'off');
                I_plot = imshow(part.icon, 'Parent', curr_axes);
                
                % Plot part bbox on icon.
                hold(curr_axes, 'on');

                plot(part.icon_bbox(:, [1 3 3 1 1])', part.icon_bbox(:, [2 2 4 4 2])', 'b', 'Parent', curr_axes)
                
                
                % Set the plots' callbacks (the previous callbacks are
                % removed when the new images are plotted).
                set(I_plot, 'ButtonDownFcn', removeButtonHandlers{i});
            end
        end
        
        % Enable/disable the corresponding navigation�button
        % if we're on the first or last page.
        if parts_curr_page == 1
            set(parts_prev_button, 'Enable', 'off');
        else
            set(parts_prev_button, 'Enable', 'on');
        end
        
        if parts_curr_page == get_parts_num_pages()
            set(parts_next_button, 'Enable', 'off');
        else
            set(parts_next_button, 'Enable', 'on');
        end
        
        % Update the current page display.
        set(parts_nav_text, 'String', sprintf('%i/%i', parts_curr_page, get_parts_num_pages()));
    end

    function add_part(part)
        curr_parts{end+1} = part;
        redraw_parts_display();
    end

    function parts_num_pages = get_parts_num_pages
        parts_num_pages = max(1, ceil(length(curr_parts)/parts_per_page));
    end

    function partsPrevButtonHandler(~, ~)
        % Callback for when the user requests the previous page.
        if parts_curr_page > 1
            parts_curr_page = parts_curr_page - 1;
            redraw_parts_display();
        end
    end

    function partsNextButtonHandler(~, ~)
        % Callback for when the user requests the next page.
        
        if parts_curr_page < get_parts_num_pages()
            parts_curr_page = parts_curr_page + 1;
            redraw_parts_display();
        end
    end

    function redraw_object_display()
        % Update the object display and navigation display.
        
        % Fill each tile with the corresponding object.
        for i = 1:object_ims_per_page
            curr_selection_button = object_selection_buttons(i);
            % Translate from the grid indices to the ims indices.
            ims_i = ((object_curr_page-1)*object_ims_per_page)+i;
            
            % Get a reference to the current tile's axes.
            curr_axes = object_axes(i);
            
            if ims_i > ims_length
                % Hide tiles if there aren't enough pictures to fill them.
                set(object_tiles(i), 'BackgroundColor', get(object_grid, 'BackgroundColor'));
                cla(curr_axes);
                set(curr_axes, 'Visible', 'off');
                set(curr_selection_button, 'Visible', 'off');
            else
                % Show tile in case it was hidden before.
                set(object_tiles(i), 'BackgroundColor', [0 0 0], 'ButtonDownFcn', selectionButtonHandlers{i});
                set(object_selection_buttons(i), 'Visible', 'on');
                
                % Show the image for the object.
                I_plot = imshow(I{ims_i},'Parent', curr_axes, 'Border', 'tight');
                
                % Set the plots' callbacks (the previous callbacks are
                % removed when the new images are plotted).
                set(I_plot, 'ButtonDownFcn', selectionButtonHandlers{i});
                
                % redraw_object_display selection button.
                redraw_button(i);
            end
        end
        
        % Enable/disable the corresponding navigation�button
        % if we're on the first or last page.
        if object_curr_page == 1
            set(object_prev_button, 'Enable', 'off');
        else
            set(object_prev_button, 'Enable', 'on');
        end
        
        if object_curr_page == object_num_pages
            set(object_next_button, 'Enable', 'off');
        else
            set(object_next_button, 'Enable', 'on');
        end
        
        % Update the current page display.
        set(object_nav_text, 'String', sprintf('%i/%i', object_curr_page, object_num_pages));
    end

    function redraw_button(i)
        % Draw a button, given its grid index.
        
        % Translate from the grid index to the ims index.
        ims_i = ((object_curr_page-1)*object_ims_per_page) + i;
        
        % Get a reference to the current button.
        curr_selection_button = object_selection_buttons(i);
        
        % Ensure that the button is visible.
        set(curr_selection_button, 'Visible', 'on');
    end

    function objectPrevButtonHandler(~, ~)
        % Callback for when the user requests the previous page.
        if object_curr_page > 1
            object_curr_page = object_curr_page - 1;
            redraw_object_display();
        end
    end

    function objectNextButtonHandler(~, ~)
        % Callback for when the user requests the next page.
        
        if object_curr_page < object_num_pages
            object_curr_page = object_curr_page + 1;
            redraw_object_display();
        end
    end

    function selectionButtonHandler = createSelectionButtonHandler(i)
        % Create a callback for the click action on a given grid index.
        
        function f(~, ~)
            ims_i = ((object_curr_page-1)*object_ims_per_page)+i;
            part.im = ims{ims_i};
            part.obj_bbox = obj_bboxes{ims_i};
            
            % Get the part's bounding box from the user.
            part.bbox = ui_select_bbox(I{ims_i});
            
            % Quit if user didn't select a bbox.
            if isempty(part.bbox)
                return;
            end
            
            % Account for the offset from the object box.
            part.bbox = [part.bbox(1)+part.obj_bbox(1) part.bbox(2)+part.obj_bbox(2) part.bbox(3)+part.obj_bbox(1) part.bbox(4)+part.obj_bbox(2)] - crop_padding;
            
            if select_features_mask
                % Add the part's highlighted mask.
                mask_I = convert_to_I(part.im);
                part_crop_padding = 0;  % Don't allow drawing features directly on the crop edges.
                m = ui_freehand_draw(crop(mask_I, part.bbox, part_crop_padding));
                if ~isempty(m)
                    part.features_mask = m;
                    % Pad the mask to fit in the whole image.
                    m = zeros(size(mask_I, 1), size(mask_I, 2));
                    m(part.bbox(2)-part_crop_padding:part.bbox(4)+part_crop_padding, part.bbox(1)-part_crop_padding:part.bbox(3)+part_crop_padding) = part.features_mask;
                    part.features_mask = m;
                end
            end
            
            % Add the icon, a small image representative of what this part looks like.
            part.icon = convert_to_I(part.im);
            
            if isfield(part, 'features_mask')
                % Draw the highlighted features.
                part.icon(:, :, 1) = part.icon(:, :, 1) + part.features_mask;
                part.icon(:, :, 2) = part.icon(:, :, 2) - part.features_mask;
                part.icon(:, :, 3) = part.icon(:, :, 3) - part.features_mask;
            end
            
            % Crop the icon to the object.
            part.icon = crop(part.icon, part.obj_bbox, crop_padding);
            
            % Add the part's bbox in the icon's coordinate system.
            part.icon_bbox = part.bbox + crop_padding;
            part.icon_bbox = [part.icon_bbox(1)-part.obj_bbox(1) part.icon_bbox(2)-part.obj_bbox(2) part.icon_bbox(3)-part.obj_bbox(1) part.icon_bbox(4)-part.obj_bbox(2)];
            
            % Add the part.
            add_part(part)
        end
        selectionButtonHandler = @f;
    end

    function removeButtonHandler = createremoveButtonHandler(i)
        % Create a callback for the click action on a given remove button.
        
        function f(~, ~)
            curr_parts(((parts_curr_page-1)*parts_per_page)+i) = [];
            redraw_parts_display();
        end
        removeButtonHandler = @f;
    end

    function finishButtonHandler(~, ~)
        parts = curr_parts;
        close(window);
    end
end

function cropped_I = crop(I, bbox, padding)
% Return the cropped image.

cropped_bbox = get_cropped_bbox(I, bbox, padding);
cropped_I = I(cropped_bbox(2):cropped_bbox(4), cropped_bbox(1):cropped_bbox(3), :);
end

function cropped_bbox = get_cropped_bbox(I, bbox, padding)
% Return the bounding box of the crop.

[height width ~] = size(I);
cropped_bbox = [max(1, bbox(1)-padding) max(1, bbox(2)-padding) min(width, bbox(3)+padding) min(height, bbox(4)+padding)];
end

function bbox = ui_select_bbox(im)
% bbox = ui_select_bbox(im)
%
% Get the user's bounding box selection from an image.
%
% Input:
%   im: an image or path to an image for the user to select a bbox from
%

im = convert_to_I(im);
im_width = size(im, 2);
im_height = size(im, 1);

% Keep track of current click state.
%   0 - enter top-left coordinate
%   1 - enter top-right coordinate
%   2 - wait for confirmation/restart
state = 0;

bbox = [];
x1 = 0;
y1 = 0;
x2 = 0;
y2 = 0;

scrsz = get(0,'ScreenSize');

    function fKeyHandler(~, e)
        switch e.Key
            case 'escape';
                close(f);
            case 'r'
                restartButtonHandler();
            case 'f'
                confirmButtonHandler();
        end
    end

    function fButtonDownHandler(~, ~)
        pointSelected();
    end

    function pointSelected()
        point = get(im_axes, 'CurrentPoint');
        switch state
            case 0
                x1 = point(1);
                y1 = point(3);
                if x1 < 1 || y1 < 1 || x1 > im_width || y1 > im_height
                    waitfor(warndlg('Point must be in image.', 'Out of Bounds', 'modal'));
                else
                    hold off; imshow(im, 'Parent', im_axes);
                    hold on; plot(x1, y1, '*', 'Parent', im_axes);
                    plot([x1 x1], [y1 im_height]);
                    plot([x1 im_width], [y1 y1]);
                    state = 1;
                end
            case 1
                x2 = point(1);
                y2 = point(3);
                if x2 <= x1 || y2 <= y1
                    waitfor(warndlg('Point must be below and to the right of the first point.', 'Out of Bounds', 'modal'));
                elseif x2 > im_width || y2 > im_height
                    waitfor(warndlg('Point must be in image.', 'Out of Bounds', 'modal'));
                else
                    hold off; imshow(im, 'Parent', im_axes);
                    hold on; plot([x1 x2], [y1 y2], '*', 'Parent', im_axes);
                    plot([x1 x2 x2 x1 x1], [y1 y1 y2 y2 y1]);
                    
                    state = 2;
                    set(buttons_hbox, 'Visible', 'on');
                    set(restart_button, 'Visible', 'on');
                    set(confirm_button, 'Visible', 'on');
                    set(f, 'Pointer', 'arrow');
                end
        end
    end

f = figure('Name', 'Click on the image to select a bounding box.', 'WindowStyle', 'modal', 'Pointer', 'fullcrosshair', 'WindowKeyPressFcn', @fKeyHandler, 'WindowButtonDownFcn', @fButtonDownHandler);
f_pos = get(f, 'Position');
f_pos(1) = (scrsz(3)-f_pos(1))/2;  % center horizontally
f_pos(2) = (scrsz(4)-f_pos(2))/2-50;  % center verically and move down so confirmation box doesn't occlude image
set(f, 'Position', f_pos);

main_vbox = uiextras.VBox('Parent', f);
im_axes = axes('Parent', main_vbox);
buttons_hbox = uiextras.HBox('Parent', main_vbox, 'Visible', 'off');
    function restartButtonHandler(~, ~)
        if state == 2
            set(buttons_hbox, 'Visible', 'off');
            hold off; imshow(im, 'Parent', im_axes);
            set(f, 'Pointer', 'fullcrosshair');
            state = 0;
        end
    end
restart_button = uicontrol('Parent', buttons_hbox, 'Style', 'pushbutton', 'Callback', @restartButtonHandler, 'String', 'Restart Selection (r)', 'BackgroundColor', [1 .7 .7], 'Visible', 'off');
    function confirmButtonHandler(~, ~)
        if state == 2
            bbox = round([x1 y1 x2 y2]);
            close(f);
        end
    end
confirm_button = uicontrol('Parent', buttons_hbox, 'Style', 'pushbutton', 'Callback', @confirmButtonHandler, 'String', 'Finish (f)', 'BackgroundColor', [.7 1 .7], 'Visible', 'off');

set(main_vbox, 'Sizes', [-1 20]);

hold off; imshow(im, 'Parent', im_axes);
waitfor(f);
end

function mask = ui_freehand_draw(im)

I = convert_to_I(im);
mask = [];
curr_mask = zeros(size(I, 1), size(I, 2));
is_drawing = false;

% Keep track of which tool is being used.
%   0 - draw
%   1 - erase
brush_size = 0;
eraser_size = brush_size;
tool = 0;

lastPoint = [];
pointsGroup = [];
points = {};

    function window_key_handler(~, e)
        % Callback for keyboard events in the main window.
        switch e.Key
            case 'escape'
                close(window);
            case {'space', 'return'}
                switch tool
                    case 0
                        update_tool(1);
                    case 1
                        update_tool(0);
                end
            case {'b', 'p', 'd'}
                update_tool(0);
            case {'e', 'delete', 'backspace'}
                update_tool(1);
            case 'equal'
                set(tool_size_slider, 'Value', min(get(tool_size_slider', 'Max'), brush_size+1));
                update_tool_size();
            case 'hyphen'
                set(tool_size_slider, 'Value', max(get(tool_size_slider', 'Min'), brush_size-1));
                update_tool_size();
            case 'f'
                finishButtonHandler();
            case 'r'
                restartButtonHandler()
        end
    end

    function update_tool(t)
        tool = t;
        
        % Set cursor.
        switch tool
            case 0
                P = ones(16)*NaN;
                P = MidpointCircle(P, brush_size+3, 9, 9, 1);
                P = MidpointCircle(P, brush_size+2, 9, 9, 2);
            case 1
                P = ones(16)*NaN;
                P = MidpointCircle(P, eraser_size+3, 9, 9, 2);
                P = MidpointCircle(P, eraser_size+2, 9, 9, 1);
        end
        
        set(gcf,'Pointer', 'custom', 'PointerShapeCData', P, 'PointerShapeHotSpot', [9 9])
    end

    function update_tool_size()
        brush_size = get(tool_size_slider, 'Value');
        eraser_size = get(tool_size_slider, 'Value');
        update_tool(tool)
    end

    function apply_tool(plotPoints)
        switch tool
            case 0
                value = 1;
                width = brush_size;
            case 1
                value = 0;
                width = eraser_size;
        end
        
        curr_mask = plot_to_matrix(curr_mask, plotPoints(:, 1), plotPoints(:, 2), value, width);
        
        g = I;
        g(:, :, 1) = g(:, :, 1) + curr_mask;
        g(:, :, 2) = g(:, :, 2) - curr_mask;
        g(:, :, 3) = g(:, :, 3) - curr_mask;
        imshow(g);
    end


    function startDraw(~, ~)
        startPoint = get(gca, 'CurrentPoint');
        startPoint = [ ...
            max(1, min(size(curr_mask, 2), startPoint(1))) ...
            max(1, min(size(curr_mask, 1), startPoint(3)))];
        
        lastPoint = startPoint;
        is_drawing = true;
        
        apply_tool(startPoint);
        
        pointsGroup = startPoint;
    end

    function continueDraw(~, ~)
        if is_drawing
            continuePoint = get(gca, 'CurrentPoint');
            continuePoint = [ ...
                max(1, min(size(curr_mask, 2), continuePoint(1))) ...
                max(1, min(size(curr_mask, 1), continuePoint(3)))];
            
            plotPoints = [lastPoint; continuePoint];
            apply_tool(plotPoints);
            
            pointsGroup = [pointsGroup; continuePoint];
            lastPoint = continuePoint;
        end
        
        update_tool_size();
    end

    function endDraw(~, ~)
        is_drawing = false;
        points{end+1} = pointsGroup;
    end

window = figure( ...
    'WindowKeyPressFcn', @window_key_handler, ...
    'WindowButtonDownFcn',@startDraw, ...
    'WindowButtonMotionFcn', @continueDraw, ...
    'WindowButtonUpFcn', @endDraw, ...
    'WindowStyle', 'modal', ...
    'Name', 'Click and drag to highlight important features');
set(window, 'MenuBar', 'none');

main_vbox = uiextras.VBox();
toolbar = uiextras.HBox('Parent', main_vbox);
uicontrol('Parent', toolbar, 'Style', 'pushbutton', 'String', 'Draw (d)', 'Callback', @(~, ~) update_tool(0));
uicontrol('Parent', toolbar, 'Style', 'pushbutton', 'String', 'Erase (e)', 'Callback', @(~, ~) update_tool(1));
controls = uiextras.HBox('Parent', main_vbox);
uicontrol('Parent', controls, 'Style', 'text', 'String', 'Size (+/-):');
tool_size_slider = uicontrol('Parent', controls, 'Style', 'slider', 'SliderStep', [1 1], 'Min', 0, 'Max', 4, 'Value', brush_size);
set(controls, 'Sizes', [50 -1]);
axes('Parent', main_vbox);

bottom_controls_hbox = uiextras.HBox('Parent', main_vbox);
    function restartButtonHandler(~, ~)
        curr_mask = zeros(size(I, 1), size(I, 2));
        imshow(I);
    end
uicontrol('Parent', bottom_controls_hbox, 'Style', 'pushbutton', 'String', 'Restart Drawing (r)', 'Callback', @restartButtonHandler, 'BackgroundColor', [1 .7 .7]);
    function finishButtonHandler(~, ~)
        mask = curr_mask;
        close(window);
    end
uicontrol('Parent', bottom_controls_hbox, 'Style', 'pushbutton', 'String', 'Finish (f)', 'Callback', @finishButtonHandler, 'BackgroundColor', [.7 1 .7]);


set(main_vbox, 'Sizes', [20 20 -1 20]);

% Update cursor now that figure is created.
update_tool(tool);

imshow(I);
waitfor(window);
end