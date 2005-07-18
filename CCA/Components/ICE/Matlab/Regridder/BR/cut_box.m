function [pieces,status] = cut_box(r,other_rect,opts,points_old,points_new,rect_old,rect_new,k)
%CUT_BOX Cut an overlapping box into non-overlapping boxes.
%   We look at a box R that overlaps a box collection OTHER_RECT. We break it into a box
%   collection PIECES that do not overlap OTHER_RECT. The other arguments (OPTS,...) are needed
%   only for visualization purposes (the old and new flagged cells, if this function is called
%   from UPDATE_CLUSTER).
%
%   See also BOX_INTERSECT, UPDATE_CLUSTER.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments. 

% Cut a box r that overlaps the OTHER_RECT collection into boxes
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
num_other   = size(other_rect,1);                               % Number of other rectangles (than r)

%%%%% Prepare a list of permitted shifts of the new box under consideration
d           = 1;
num_shifts  = r(d+dim)-r(d)+1;
cut_sig     = cell(num_shifts,1);
other_dim   = setdiff([1:dim],d);
other_col   = setdiff([1:2*dim],[d d+dim]);

for count = 1:num_shifts,
    cut             = r(d)+count-1;    
    cut_plane       = r;
    cut_plane([d d+dim]) = cut;
    other_intersect = box_intersect(other_rect,cut_plane);
    relevant        = find(box_volume(other_intersect) > 0);
    points_intersect= other_intersect(relevant,other_col);
    
    % The following works only in 2D, but can be extended to any-D using a lexicographic ordering of points_intersect
    points_intersect = sort(points_intersect);    
    temp    = [r(other_dim)];
    for i = 1:2:length(points_intersect)-1,
        temp = [temp; points_intersect(i)-1; points_intersect(i+1)+1];
    end
    temp    = [temp; r(other_dim+dim)];    
    
    cut_sig{count} = [];
    for i = 1:2:length(temp)-1,
        if (temp(i) <= temp(i+1))
            cut_sig{count} = [cut_sig{count}; temp(i:i+1)];
        end
    end    
    
end

change              = ones(num_shifts-1,1);
for count = 1:num_shifts-1,
    if (size(cut_sig{count}) == size(cut_sig{count+1}))
        if (min(cut_sig{count}(:) == cut_sig{count+1}(:)) == 1)
            change(count) = 0;
        end
    end
end

cut         = find(change);
%cut
num_cuts    = length(cut);
cut         = [0; cut; num_shifts];
pieces      = zeros(0,4);

% for i = 1:num_shifts
%     fprintf('cut_sig{%d} = ',i);
%     cut_sig{i}
% end
% cut

for i = 2:length(cut),
    num_cut_boxes   = length(cut_sig{cut(i)})/2;
    temp            = zeros(num_cut_boxes,2*dim);
    temp(:,d)       = cut(i-1)+r(d);
    temp(:,d+dim)   = cut(i)+r(d)-1;
    for j = 1:num_cut_boxes        
        temp(j,other_col) = cut_sig{cut(i)}(2*j-1:2*j)';       % Works only for 2D
    end
%    cut_sig{cut(i)}
%    temp
    pieces = [pieces; temp];
end

% Delete empty boxes from pieces
n   = size(pieces,1);
nopoints = ones(n,1);
for i = 1:n
    s = points_new(pieces(i,1):pieces(i,3),pieces(i,2):pieces(i,4));                      % Flag data of this box
    ind = find(s);                                                  % 2D indices of the points in this box
    nopoints(i) = isempty(ind);
end
pieces(find(nopoints),:) = [];
status = 1;
