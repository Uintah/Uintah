function [rect,status,stats] = update_cluster(points_old,rect_old,points_new,opts);
% UPDATE_CLUSTER Update a set of covering boxes. 
%    If POINTS_OLD is the old set of flagged cells, and RECT_OLD
%    is the covering box set, we add some new flagged cells POINTS_NEW, and try to update the covering
%    set. Possible output status code:
%    status = 0      We succeeded to update, and the new box set is rect.
%    status = -1     We failed tp update. Run CREATE_CLUSTER again.
%    RECT is the updated set of boxes, and the STATS struct contain statistics on the box 
%    collection RECT.
%
%    See also CREATE_CLUSTER, DISSECT_BIG_BOX, DISSECT_BOX, FIND_HOLE, FIND_INFLECTION, TEST_MOVEMENT.
 
% 
% Author: Oren Livne
% Date  : 05/12/2004    Version 1: move each single new box at a time to eliminate overlap with old+new
%         05/13/2004    Version 2: search in a local set of shifts (not Dave's brilliant spiral idea, but better here)
%         05/24/2004    Version 3: added treatment of diagonal movement: try to break a non-shifted new box into smaller boxes

%%%%%%%%%% Set and print parameters
if (nargin < 4)                                                 % Default parameters
    opts.efficiency = 0.8;                                      % Lowest box efficiency allowed
    opts.low_eff    = 0.5;                                      % Efficiency threshold for boxes that don't have holes/inflections
    opts.min_side   = 10;                                       % Minimum box side length allowed (in all directions)
    opts.max_volume = 100;                                      % Maximum box volume allowed
    opts.print      = 0;                                        % Printouts flag
    opts.plot       = 0;                                        % Plots flag
end
fprintf('<<<<<<<<<<<<<< UPDATE_CLUSTER: updating a cluster given more flagged points >>>>>>>>>>>>>>\n');
fprintf('Parameters for creating the new cluster:\n');          % Print parameters
fprintf('Efficiency threshold: %.1f%%\n',100*opts.efficiency);  % Efficiency threshold
fprintf('Min. box side length: %d\n',opts.min_side);            % Min. box size
fprintf('Max. box volume     : %d\n',opts.max_volume);          % Max. box volume

%%%%% Delete empty boxes from the old box set, in case we deleted points from points_old and some of the boxes are unusable (but might interfer with shifting the new set around later on in this routine)
efficiency      = box_efficiency(points_old,rect_old);          % Compute efficiencies of boxes
empty           = find(efficiency < 1e-13);                     % Empty boxes (efficiency = 0)
rect_old(empty,:) = [];                                         % Remove empty boxes
if ((opts.print) & (~isempty(empty)))                           % Print the boxes that were deleted
    fprintf('Empty boxes deleted:');
    fprintf('%d ',empty);
    fprintf('\n');
end

%%%%%%%%%% Initialize, find the initial new covering, get rid of non-relevant new points
big_box     = [-10000000 -10000000 10000000 10000000];          % !!!!!!!!!!!! LATER: REPLACE WITH THE DOMAIN !!!!!!!!!!!!!!!!!!
num_actions = 2;                                                % No. different actions trying to resolve a non-conforming new box
dim         = length(size(points_old));                         % Dimension of the problem
status      = 0;                                                % Default status is success, if we fail we break and status=-1
points      = points_old + points_new;                          % Merge the old and new point sets (for binary images, + = union)
points_new_orig = points_new;
for k = 1:size(rect_old,1)                                      % Get rid of new points that are covered by old boxes
    r = rect_old(k,:);                                          % Old box no. k
    points_new(r(1):r(3),r(2):r(4)) = 0;                        % Delete new points in this box
end
[i,j]       = find(points_new);                                 % Index arrays [i,j] of flagged cells 
if (isempty(i))                                                 % No new points are flagged
    rect    = rect_old;                                         % Return the old box set
    status  = 0;                                                % Return a success code
    stats   = final_stats(points,rect,opts);                    % Final printouts (overall statistics) and plot-outs
    fprintf('<<<<<<<<<<<<<< END UPDATE_CLUSTER >>>>>>>>>>>>>>\n');
    return;
end
opts_create = opts;                                             % Options for CREATE_CLUSTER function
opts_create.print = 0;                                          % No printouts from CREATE_CLUSTER, please!
opts_create.plot = 0;                                           % No plots from CREATE_CLUSTER, please!
rect_new    = create_cluster(points_new,opts_create);                  % Create a set over the new points, might overlap old set
num_new     = size(rect_new,1);                                 % Number of new rectangles
%opts.print  = 1;
%opts.plot   = 1;

%%%%% Plot-outs: plot the points and the current boxes. The considered box is in red.
if (opts.plot)
    figure(1);
    clf;
    plot_points(points_new,'red');
    hold on;
    plot_points(points_old,'black');
    plot_boxes(rect_old,'black');
    plot_boxes(rect_new,'red');
    pause
end

%%%%%%%%%% Loop over new boxes; try to move each one so it does not overlap any old or new one
nactions = num_actions*ones(num_new,1);

k       = 1;                                                    % Index of new box to be processed
while ((k <= num_new) & (~isempty(rect_old)))                   % Do until all boxes have been processed
    r               = rect_new(k,:);                            % Box coordinates
    s               = points_new(r(1):r(3),r(2):r(4));          % Flag data of this box
    sz              = box_size(r);                              % Vector containing the size of the box: [size_x,size_y]
    efficiency      = length(find(s))/prod(sz);                 % Percentage of flagged cells in s
    [a,sorted_dims] = sort(-sz);                                % Sort box sizes in descending orders
    [i,j]           = find(s);
    tight           = [min(i) min(j) max(i) max(j)] + [r(1) r(2) r(1) r(2)] - 1;
    if (opts.print)
        fprintf('Considering box #%3d at coordinates [%3d,%3d,%3d,%3d]   size = %d x %d,  vol = %d, efficiency = %f\n',k,r,sz+1,box_volume(r),efficiency);
        fprintf('Tight box = [%3d,%3d,%3d,%3d]\n',tight);
    end
    
    %%%%% Plot-outs: plot the points and the current boxes. Old is black, new is red. The considered box is in green.
    if (opts.plot)
        figure(1);
        clf;
        plot_points(points_new,'red');
        hold on;
        plot_points(points_old,'black');
        plot_boxes(rect_old,'black');
        plot_boxes(rect_new,'red');
        plot_boxes(rect_new(k,:),'green');
        pause
    end
 
    %%%%% Initialize parameters for shift loop
    other_rect  = [rect_old; rect_new(setdiff(1:num_new,k),:)]; % All the other rectangles - old+new
    overlap     = 1;                                            % In the loop below: 0 if we overlap no other rectangle, 1 if we do

    %%%%% Loop over actions to find a cut
    cut.found   = 0;                                            % Start: we don't know where to dissect the box
    for action = 1:nactions(k),                                 % Try different actions to find a dissection plane ("cut")
        if (~overlap)                                           % If box no longer overlapping any other box, accept it
            break;
        end
        switch (action)                                         % Each action is attached to a certain piece of code below, trying to eliminate overlap
        case 1
            if (opts.print)
                fprintf('Action 1: try to shift box\n');
            end
            [t,overlap] = shift_box(r,other_rect,tight,opts,...
                points_old,points_new,rect_old,rect_new,...
                k);                                             % Try to shift box around to eliminate overlap with any other box. The shifted box (if we succeed) is t.
        case 2
            if (opts.print)
                fprintf('Action 2: try to break into smaller boxes and shift them\n');
            end
            pieces = cut_box(r,other_rect,opts,...
                points_old,points_new,rect_old,rect_new,...
                k);                                             % Try to shift box around to eliminate overlap with any other box. The shifted box (if we succeed) is t.
                        
            %%%%% Plot-outs: plot the points and the current boxes. Old is black, new is red. The considered box is in green.
            if (opts.plot)
                figure(1);
                clf;
                plot_points(points_new,'red');
                hold on;
                plot_points(points_old,'black');
                plot_boxes(rect_old,'black');
                plot_boxes(rect_new,'red');
                plot_boxes(rect_new(k,:),'green');
                plot_boxes(pieces,'blue');
                pause
            end

            pieces = box_extend(pieces,big_box,opts.min_side);

            %%%%% Plot-outs: plot the points and the current boxes. Old is black, new is red. The considered box is in green.
            if (opts.plot)
                figure(1);
                clf;
                plot_points(points_new,'red');
                hold on;
                plot_points(points_old,'black');
                plot_boxes(rect_old,'black');
                plot_boxes(rect_new,'red');
                plot_boxes(rect_new(k,:),'green');
                plot_boxes(pieces,'blue');
                pause
            end
            overlap     = 2;
        end
    end
       
    switch (overlap)
    case 0
        if (opts.print)
            fprintf('Shift found, accepting and going to the next box\n');
        end
        rect_new(k,:) = t;                                      % Replace the kth new rectangle with the shifted one, t
        k = k+1;                                                    % Consider the next new rectangle        
    case 1
        if (opts.print)
            fprintf('No shift found, giving up\n');
        end
        rect    = [];                                           % Return an empty results
        status  = -1;                                           % Negative status = UPDATE_CLUSTER failed
        stats   = [];
        fprintf('<<<<<<<<<<<<<< END UPDATE_CLUSTER >>>>>>>>>>>>>>\n');
        return;
    case 2
        if (opts.print)
            fprintf('Box broken into smaller pieces\n');
        end
        rect_new        = [rect_new; pieces];                               % Add the two halves to the list
        rect_new(k,:)   = [];                                       % Delete box k from the list, so now k points to the "next box" to be considered
        num_new         = size(rect_new,1);                                 % Number of new rectangles        
        nactions        = [nactions; 1*ones(size(pieces,1),1)];
        nactions(k)     = [];
    end
    if (opts.print)
        fprintf('\n');
    end
end

%%%%% Check if new boxes are inside the domain; if not then fail to update (note: no need to check that they are in the domain before this point in the code)
domain  = [repmat(1,1,dim) size(points)];
for k = 1:size(rect_new)
    if (~is_box_subset(domain,rect_new(k,:)))
        if (opts.print)
            fprintf('New box not within the domain, giving up\n');
        end
        rect    = [];                                           % Return an empty results
        status  = -1;                                           % Negative status = UPDATE_CLUSTER failed
        stats   = [];
        fprintf('<<<<<<<<<<<<<< END UPDATE_CLUSTER >>>>>>>>>>>>>>\n');
        return;
    end
end

%%%%% Unify old and new sets, and delete empty boxes
rect            = [rect_old; rect_new];                         % Merge the old and new box sets
efficiency      = box_efficiency(points,rect);                  % Compute efficiencies of boxes
empty           = find(efficiency < 1e-13);                     % Empty boxes (efficiency = 0)
rect(empty,:)   = [];                                           % Remove empty boxes
if ((opts.print) & (~isempty(empty)))                           % Print the boxes that were deleted
    fprintf('Empty boxes deleted:');
    fprintf('%d ',empty);
    fprintf('\n');
end

stats = final_stats(points,rect,opts);                          % Final printouts (overall statistics) and plot-outs
fprintf('<<<<<<<<<<<<<< END UPDATE_CLUSTER >>>>>>>>>>>>>>\n');
