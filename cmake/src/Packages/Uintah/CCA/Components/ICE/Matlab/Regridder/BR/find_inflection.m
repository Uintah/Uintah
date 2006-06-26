function cut = find_inflection(r,sig,min_side)
%FIND_INFLECTION Look for an inflection point in a box to dissect it.
%   We Look for an inflection point (a zero-crossing in the second derivative of the signature),
%   given the signature arrays SIG and the minimum side length MIN_SIDE. R is the box to be dissected.
%   The output CUT is the standard structure for a cut (if we don't find an inflection point, CUT.FOUND = 0).
%
%   See also CREATE_CLUSTER, COMPUTE_SIGNATURES, DISSECT_BOX, FIND_HOLE.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

cut.found   = 0;                                                % Default: no cut found
sz          = box_size(r);                                      % Rectangle side lengths
dim         = size(sig,1);                                      % Dimension of the problem
delta       = cell(dim,1);                                      % Second-derivative-of-signature in all dimensions
best_place  = -ones(dim,1);                                     % Absolute coordinate of best place to cut
value       = -ones(dim,1);                                     % Sharpness value of edge; -1 = dummy (no allowed edge found)
for d = 1:dim                                                   % Loop over dimensions
    delta{d} = diff(diff(sig{d}));                              % Discrete second-derivative
    schange_abs = abs(diff(delta{d}));                          % Gradient absolute value
    schange = zeros(size(schange_abs));                         % Array of flags of sign changes in delta
    for i = 1:length(delta{d})-1                                % Loop over the delta array in this direction
        schange(i) = sign(delta{d}(i)*delta{d}(i+1));           % sign change from i to i+1: 1, none; 0, one of them is zero (so sign change); -1, sign change
    end
    len         = length(sig{d});                               % Length of box in this dimension
    zero_cross  = find(schange <= 0);                           % Indices i for which delta changes sign (i -> i+1)
    zero_cross  = intersect(zero_cross,...
        [min_side:len-min_side]-1);                             % Do not allow zero crossing that are too close to the boundaries, because we need to keep a minimum side length. -1 because we move to delta coordinate, [2..len-1]                
    if (~isempty(zero_cross))                                   % If there exist zero crossing...
        edge            = schange_abs(zero_cross);              % Save only the relevant indices (zero_cross) in schange_abs
        max_cross_value = max(edge);                            % Find the sharpest edge value
        max_cross       = zero_cross(...
            find(edge == max_cross_value)) + 1;                 % Find where are the sharpest edge locations; +1 because delta is defined on [2..len-1] and we convert here back to sig coordinate
        center          = len/2+0.5;                            % Center coordinate of this direction (origin at 1)
        distance = abs(max_cross+0.5-center);                   % Distance of zero-crossings for the center
        best_cross = max_cross(...
            find(distance == min(distance)));                   % Find those closest to the center
        best_place(d) = best_cross(1) + r(d)-1;                 % Covert back to absolute coordinates and save in best_place
        value(d) = max_cross_value;                             % Save edge value for comparison between dimensions
    end
end
zero_cross_dims = find(value >= 0);                             % All dims for which there exists a zero crossing
if (~isempty(zero_cross_dims))                                  % If there exists a zero crossing
    %                 best_place
    %                 value
    max_cross_dims  = find(value == max(value));                % Find sharpest edge(s)
    best_dims       =  max_cross_dims(find(...
        sz(max_cross_dims) == max(sz(max_cross_dims))));        % Find the sharpest edge(s) in the longest direction(s)
    cut.found       = 1;                                        % Found a cut, flag up
    cut.dim         = best_dims(1);                             % Dimension along which we cut
    cut.place       = best_place(cut.dim);                      % Coordinate of cut along that dimension
    fprintf('Inflection point found\n');                        % Printout
end
