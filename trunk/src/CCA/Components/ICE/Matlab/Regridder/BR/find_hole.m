function cut = find_hole(r,sig,sorted_dims,min_side)
%FIND_HOLE Look for a hole in a box to dissect it.
%   If a signature has a zero value, we know we can dissect the box along this plane.
%   This is used in the clustering algorithm (CREATE_CLUSTER) to recursively dissect big
%   or inefficient boxes.
%   The algorithm in this function look for a hole, given the signature arrays sig and the minimum side length min_side.
%   SORTED_DIMS specifies the order by which we loop over the dimensions when we look for holes.
%   MIN_SIDE is the minimum allowed side length for a box: we don't cut too close to the boundaries of the original
%   box to keep this minimum side length after dissection. R is the box to be dissected.
%   The output CUT is the standard structure for a cut (if we don't find a hole, CUT.FOUND = 0).
%
%   See also CREATE_CLUSTER, COMPUTE_SIGNATURES, DISSECT_BOX.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

cut.found   = 0;                                                % Default: no cut found
for d = sorted_dims                                             % Loop over dimensions in decreasing box size
    len     = length(sig{d});                                   % Length of box in this dimension
    hole    = find(sig{d} == 0);                                % Look for holes in direction d
    hole    = intersect(hole,[min_side:len-min_side]);          % Do not allow holes that are too close to the boundaries, because we need to keep a minimum side length
    if (~isempty(hole))                                         % If found hole...
        center      = len/2+0.5;                                % Center coordinate of this direction (origin at 1)
        distance    = abs(hole+0.5-center);                     % Distance of holes from the center
        best_hole   = hole(find(distance == min(distance)));    % Find the holes closest to the center
        cut.found   = 1;                                        % Found a cut, flag up
        cut.place   = best_hole(1)+r(d)-1;                      % Choose one of them and convert back into absolute coordinates
        cut.dim     = d;                                        % Dimension along which we cut
        fprintf('Found hole\n');                                % Printout
        break;                                                  % We terminate the loop over dimensions when we find a legal hole
    end
end
