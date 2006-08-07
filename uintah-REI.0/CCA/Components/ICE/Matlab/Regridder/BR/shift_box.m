function [t,overlap] = shift_box(r,other_rect,tight,opts,points_old,points_new,rect_old,rect_new,k)
%SHIFT_BOX Shift a box around to eliminate overlap with other boxes.
%   When a new box r overlaps any other box (old or new) in the collection OTHER_RECT, try to shift it to eliminate the overlap,
%   by an exhaustive search (sorted by shift magnitude). If we found a shift, we output the shifted box coordinates as t,
%   and overlap = 0. Otherwise, overlap = 1 and we fail (t = the original box r). t has to contain tight, which is the tight
%   bounding box around the relevant flagged cells in r.
%
%   See also UPDATE_CLUSTER.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

if (nargin < 4)
    opts.print  = 0;
    opts.plot   = 0;
end

%%%%% Initialize parameters for shift loop
dim         = length(r)/2;                                      % Dimension of the problem
overlap     = 1;                                                % In the loop below: 0 if we overlap no other rectangle, 1 if we do
within_range= 0;                                                % In the loop below: Check flag - if t contains the tight box around flagged cells (<==> t contains all flagged cells it needs to)
t           = r;                                                % t is the new (shifted) box; init t to the old box r
action      = 0;                                                % Counter of shifts, action=1 is original box (shift=(0,...,0)).
num_new     = size(rect_new,1);                                 % Number of new rectangles

%%%%% Prepare a list of permitted shifts of the new box under consideration
shift_line  = cell(dim,1);                                  % Permitted shift ranges along the different dimensions
for d = 1:dim
    shift_line{d} = [tight(d+dim)-t(d+dim):tight(d)-t(d)];  % Permitted shift ranges along dimension d, so that t still contains tight
end
[shift_cell{1:dim}] = ndgrid(shift_line{:});                % Prepare a dim-D list of the possible shifts
shift = [];                                                 % 1D list of dim-coordinates of the shifts
for d = 1:dim                                               % Loop over dimensions
    shift = [shift shift_cell{d}(:)];                       % Concatenate the d-coordinate of all shifts to the big list
end
dist        = sum(shift.^2,2);                              % Distance from original rectangle
[temp,ind]  = sort(dist);                                   % Sort by ascending distance
shift       = shift(ind,:);                                 % Apply permutation to the shift list

%%%%% Main loop over shifts: trying to find a shift for which there's no overlap and we still cover the new points
for action = 1:size(shift,1),                               % Loop over all permitted shifts
    if ((~overlap) & (within_range))                        % If there's no overlap, accept this rectangle, otherwise, try other actions
        break;
    end
    s       = shift(action,:);                              % Current shift
    if (opts.print)
        fprintf('Action %5d: shift = (%d,%d)',action,s);
    end
    within_range    = is_box_subset(t,tight);               % Check if t contains the tight box around flagged cells (<==> t contains all flagged cells it needs to)
    t(1:dim)        = r(1:dim) + s;                         % Shift box lower-left coordinate by s
    t(dim+[1:dim])  = r(dim+[1:dim]) + s;                   % Shift box upper-right coordinate by s
    within_range    = is_box_subset(t,tight);               % Check if we still cover what we should cover
    overlap         = max(is_box_intersect(other_rect,t));  % 0 if we overlap no other rectangle, 1 if we do
    if (opts.print)
        fprintf('   overlap = %d\n',overlap);
    end
%     if (opts.plot)
%         figure(1);
%         clf;
%         plot_points(points_new,'red');
%         hold on;
%         plot_points(points_old,'black');
%         plot_boxes(rect_old,'black');
%         plot_boxes(rect_new(setdiff(1:num_new,k),:),'red');
%         plot_boxes(t,'green');
%         pause
%     end
end                                                         % Failure: overlap remains 1 and t is the original box
