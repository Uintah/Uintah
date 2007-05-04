function [rect,stats] = create_cluster(points,opts);
% CREATE_CLUSTER Implementation of the Berger-Rigoustos clustering algorithm.
%    Given a set of flagged cells, we find a set of boxes. The boxes contain all the points
%    and do not overlap. They are aimed to cover the least "extra space" (non-flagged points).
%    See Berger, M. and Rigoutos, I., An algorithm for point clustering and grid generation, IEEE Trans.
%    Sys. Man Cyber., Vol. 21, No. 5, September/October 1991, pp. 1278-1286.
%
%    See also DISSECT_BIG_BOX, DISSECT_BOX, FIND_HOLE, FIND_INFLECTION, TEST_MOVEMENT, UPDATE_CLUSTER.

% Author: Oren Livne
% Date  : 04/29/2004    Version 1: basic B-R algorithm
%         05/03/2004    Version 2: added some comments
%         05/10/2004    Version 3: added min. box size control, maximal control size

%%%%%%%%%% Set and print parameters
if (nargin < 2)                                                 % Default parameters
    opts.efficiency = 0.8;                                      % Lowest box efficiency allowed
    opts.low_eff    = 0.5;                                      % Efficiency threshold for boxes that don't have holes/inflections
    opts.min_side   = 4;                                        % Minimum box side length allowed (in all directions)
    opts.max_volume = 100;                                      % Maximum box volume allowed
    opts.print      = 0;                                        % Printouts flag
    opts.plot       = 0;                                        % Plots flag
end

fprintf('<<<<<<<<<<<<<< CREATE_CLUSTER: creating a cluster from flagged points >>>>>>>>>>>>>>\n');
if (opts.print)
    fprintf('Parameters:\n');                                   % Print parameters
    fprintf('Efficiency thresh.: %.1f%%\n',100*opts.efficiency);% Efficiency threshold
    fprintf('Min. box side len : %d\n',opts.min_side);          % Min. box size
    fprintf('Max. box volume   : %d\n',opts.max_volume);        % Max. box volume
end

%%%%%%%%%% Initialize and find the first box
num_actions = 4;                                                % No. different actions to try on a box for dissecting it
dim         = length(size(points));                             % Dimension of the problem
[i,j]       = find(points);                                     % Index arrays [i,j] of flagged cells 
if (isempty(i))                                                 % No points are flagged
    rect    = [];                                               % Return an empty box set
    stats   = [];
    fprintf('<<<<<<<<<<<<<< END CREATE_CLUSTER >>>>>>>>>>>>>>\n');
    return;
end
rect        = [min(i) min(j) max(i) max(j)];                    % Bounding box for flagged cells
for d = 1:dim                                                   % If too small in any dimension, extend bounding box to the minimal side length
    lc          = d;                                            % Index of most-left point of the box in direction d
    rc          = lc+dim;                                       % Index of most-right point of the box in direction d
    if (rect(rc)-rect(lc)+1 < opts.min_side)                    % If size smaller than minimum size ...
        rect(rc) = rect(lc) + opts.min_side - 1;                % Extend right-most point so that the size = minimum size
    end
    if (rect(lc) < 1)                                           % If we exceed left boundary, shift to the right (the domain should is assumed to be big enough to allow this shift)
        rect([lc rc]) = rect([lc rc]) + 1-rect(lc);
    end
    if (rect(rc) > size(points,d))                              % If we exceed right boundary, shift to the left (the domain should is assumed to be big enough to allow this shift)
        rect([lc rc]) = rect([lc rc]) + size(points,d)-rect(rc);
    end
end

%%%%%%%%%% Main algorithm: loop over boxes, process them, and possiblly add more boxes
k       = 1;                                                    % Index of box to be processed
while (k <= size(rect,1))                                       % Do until all boxes have been processed
    r               = rect(k,:);                                % Box coordinates
    s               = points(r(1):r(3),r(2):r(4));              % Flag data of this box
    sz              = box_size(r);                              % Vector containing the size of the box: [size_x,size_y]
    efficiency      = length(find(s))/prod(sz);                 % Percentage of flagged cells in s
    [a,sorted_dims] = sort(-sz);                                % Sort box sizes in descending orders
    sig             = compute_signatures(s);                    % Compute signatures        
    if (opts.print)
        fprintf('Considering box #%3d at coordinates [%3d,%3d,%3d,%3d]   size = %d x %d,  vol = %d, efficiency = %f\n',k,r,sz,box_volume(r),efficiency);
    end
    
    %%%%% Plot-outs: plot the points and the current boxes. The considered box is in red.
    if (opts.plot)
        figure(1);
        clf;
        plot_points(points);
        hold on;
        plot_boxes(rect);
        offset = 0.2;
        h = rectangle('Position',[rect(k,1:2)-offset,[rect(k,3:4)-rect(k,1:2)]+2*offset]);
        set(h,'EdgeColor','Red');
        set(h,'LineWidth',2);
        pause
    end
    
    %%%%% Loop over actions to find a cut
    cut.found   = 0;                                            % Start: we don't know where to dissect the box
    for action = 1:num_actions,                                 % Try different actions to find a dissection plane ("cut")
        switch (action)                                         % Each action is attached to a certain piece of code below
        case 1
            if (opts.print)
                fprintf('Action 1: check efficiency\n');
            end
            if (efficiency >= opts.efficiency)                  % box efficient, but check if it's too big
                if (opts.print)
                    fprintf('Box has the required efficiency\n');
                end
                if (box_volume(r) > opts.max_volume)            % Box too big ...
                    if (opts.print)
                        fprintf('But box too big, dissect it\n');
                    end
                    cut.dim = sorted_dims(1);                   % Longest dimension
                    cut     = dissect_big_box(points,r,...
                        sig,cut.dim,opts.min_side);             % Dissect it
                end
                if (~cut.found)                                 % If we didn't cut (i.e. if box not too big), accept rectangle
                    break;
                end
            end
        case 2
            if (opts.print)
                fprintf('Action 2: look for holes\n');
            end
            cut = find_hole(r,sig,sorted_dims,opts.min_side);   % Look for a hole
        case 3
            if (opts.print)
                fprintf('Action 3: look for inflection points\n');
            end
            cut = find_inflection(r,sig,opts.min_side);         % Look for an inflection point
        case 4                                                  % No holes or inflection points; base box acceptance on its efficiency; bisect if not efficient enough
            if (opts.print)
                fprintf('Action 4: no holes or inflections, dissect if not efficienct and not too small; otherwise, check if too big\n');
            end
            cut.dim             = sorted_dims(1);               % Longest dimension
            if (efficiency <= opts.low_eff)                     % If box not efficient (efficiency <= 50% - the diagonal black case included, = 50%)
                if (sz(cut.dim) >= 2*opts.min_side)             % Bissect only if the halves are still larger than the minimum side permitted
                    cut.found       = 1;                        % We will cut, this time - bisect
                    cut.place       = floor(...
                        sz(cut.dim)/2)+r(cut.dim)-1;            % The middle absolute coordinate of this dimension
                    if (opts.print)
                        fprintf('Bisecting box because efficiency = %f < %f and side size = %d > min_side\n',efficiency,opts.low_eff,sz(cut.dim));
                    end
                end
            else                                                % Efficiency > 50%
                if (box_volume(r) > opts.max_volume)            % Box too big ...
                    cut = dissect_big_box(points,r,...
                        sig,cut.dim,opts.min_side);             % Dissect it
                end
            end
        end
        if (cut.found) 
            break;
        end
    end
    
    %%%%% Make the cut: replace the current box by its two "halves"
    if (cut.found)
        %%%%% Plot-outs and printouts: plot the cutting plane in green
        if (opts.print)
            fprintf('This box is dissected at cut.dim = %d, cut.place = %3d\n',cut.dim,cut.place);                               
        end
        if (opts.plot)
            if (cut.dim == 1)                                   % This code is specific for 2D, need to generalize to dim-D later
                h = line([cut.place cut.place]+0.5,[r(2)-0.2,r(4)+0.2]);
            else
                h = line([r(1)-0.2,r(3)+0.2],[cut.place cut.place]+0.5);
            end
            set(h,'LineWidth',3);
            set(h,'Color','Green');
        end
        
        rn          = dissect_box(points,r,cut,opts.min_side);
        rect        = [rect; rn];                               % Add the two halves to the list
        rect(k,:)   = [];                                       % Delete box k from the list, so now k points to the "next box" to be considered
        if (opts.plot)
            pause
        end
    else
        if (opts.print)
            fprintf('This box is accepted\n');
        end
        k = k+1;                                                % Couldn't find a cut accept this box and consider the next box on the list
    end
    if (opts.print)
        fprintf('\n');
    end
end

stats = final_stats(points,rect,opts);                          % Final printouts (overall statistics) and plot-outs
fprintf('<<<<<<<<<<<<<< END CREATE_CLUSTER >>>>>>>>>>>>>>\n');
