function cut = dissect_big_box(points,r,sig,d,min_side)
%DISSECT_BIG_BOX Dissect a big box into two smaller boxes because of maximum size constraint.
%   When the box R is larger than the maximum volume, we break it down into two smaller boxes
%   by calling DISSECT_BIG_BOX(POINTS,R,SIG,D,MIN_SIDE). Here POINTS is the array of flagged 
%   cells that R must cover, SIG is the signature cell array, D is the dimension along which we dissect,
%   MIN_SIDE is the minimum side length permitted. The output is a structure CUT, where
%   CUT.DIM = dimension along which we try to cut (usually the longest), and CUT.PLACE is the coordinate
%   at which we dissect R. CUT.FOUND is set to 1, indicating that we found a place to cut R at (see CREATE_CLUSTER).
%
%   See also CREATE_CLUSTER, COMPUTE_SIGNATURES, DISSECT_BOX, UPDATE_CLUSTER.

% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments. 

cut.found   = 0;                                                % Default: no cut found
dim         = size(sig,1);                                      % Dimension of the problem
len         = length(sig{d});                                   % Length of this side
eff         = zeros(2,len-1);                                   % Efficiency of the two halves of the prospective cut coordinate
rn          = cell(2,1);                                        % The two halves
rn{1}       = r;                                                % Create the "left half"
rn{2}       = r;                                                % Create the "right half"
for i = 1:len-1                                                 % This could be done more efficiently in a sliding window manner
    rn{1}(d+dim) = r(d)-1+i;                                    % The "left half": change the end index in the d-direction to the cut index
    rn{2}(d)     = r(d)-1+i+1;                                  % The "right half": change the beginning index in the d-direction to the cut index
    for h = 1:2
        s = points(rn{h}(1):rn{h}(3),rn{h}(2):rn{h}(4));        % Flag data of this box; specific for 2D
        eff(h,i) = length(find(s))/(rn{h}(d+dim)-rn{h}(d)+1);
    end
end
ratio       = max(eff,[],1)./min(eff,[],1);
ratio_inner = ratio(min_side:len-min_side);
best        = find(ratio_inner == min(ratio_inner))+min_side-1;
center      = len/2+0.5;                                        % Center coordinate of this direction (origin at 1)
distance    = abs(best+0.5-center);                             % Distance of zero-crossings for the center
best_center = best(find(distance == min(distance)));            % Find those closest to the center
cut.found   = 1;                                                % Found a cut, flag up
cut.place   = best_center(1)+r(d)-1;
cut.dim     = d;
