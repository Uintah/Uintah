function rn = box_extend(rn,r,min_side)
%BOX_EXTEND Extend box to minimum allowed size.
%   RN = BOX_EXTEND(RN,R,MIN_SIDE) updates the box collection RN. Each box
%   RN(k,:) is extended so that the minimum side length is at least MIN_SIDE.
%   It is assumed that RN is a (possibly partial) partition of an original box R,
%   and we keep sure that after extending the RN-boxes, we shift them back so
%   that they are still contained in R.
%   
%   See also CREATE_CLUSTER, DISSECT_BOX.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: handles RECT kx4 arrays, not just a single box

n   = size(rn,1);                                               % Number of boxes in rn
dim = size(rn,2)/2;                                             % Dimension of the problem

for i = 1:n                                                     % Loop over all boxes in RN
    for d = 1:dim                                               % Loop over dimensions, extend RN to a minimum side length in each d
        lc          = d;                                        % left d-coordinate index
        rc          = lc+dim;                                   % right d-coordinate index
        slack       = min_side - (rn(i,rc)-rn(i,lc)+1);         % If we're too thin, slack is positive
        if (slack > 0)                                          % If we're too thin, extend and possible shift
            ext_left  = floor(slack/2);                         % How much to extend tight bounding box on the left
            ext_right = slack - ext_left;                       % How much to extend tight bounding box on the right
            rn(i,lc) = rn(i,lc) - ext_left;                     % Left extend
            rn(i,rc) = rn(i,rc) + ext_right;                    % Right extend
            if (rn(i,lc) < r(lc))                               % If we are popping out of r to the left, shift right
                rn(i,[lc rc]) = rn(i,[lc rc]) + repmat(r(lc) - rn(i,lc),1,2);
            end
            if (rn(i,rc) > r(rc))                               % If we are popping out of r to the right, shift left
                rn(i,[lc rc]) = rn(i,[lc rc]) + repmat(r(rc) - rn(i,rc),1,2);
            end
        end
    end
end
