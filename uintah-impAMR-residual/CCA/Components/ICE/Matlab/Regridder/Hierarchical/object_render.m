function points = object_render(t,k)
%OBJECT_RENDER Position objects in the binary image of flagged cells, at time t.
%
%   POINTS = OBJECT_RENDER(TIN,T,K) positions the object list specified by TIN
%   (TIN includes their initial locations and time-movements), at time T and at
%   level k.
%   The output is a binary image POINTS of flagged and non-flagged cells (objects
%   are in fact groups of flagged cells).
%   
%   See also ADD_OBJECT, TEST_CASE, TEST_MOVEMENT.

% Author: Oren Livne
%         05/28/2004    Version 1: Created
%         06/17/2004    Replaced binary image of flagged cells by lists; changed center
%                       of object definition for 'ball', and object format for 'box'.

global_params;
dim             = length(L{k}.cell_size);                           % Dimension of the problem
points          = zeros(0,dim);                                     % Empty list of flagged cells
a               = cell(dim,1);                                      % Used to find the x and y coordinate (a{1} and a{2}, respectively) of an object

for n = o:tin.num_objects+o-1,                                      % Loop over objects
    object      = tin.object{n,1};                                  % Object handle
    move        = tin.object{n,2};                                  % Movement handle

    %%%%% Compute the center of the object at time t in cells of the coarsest level
    switch (move.type)                                              % Move the object according to its movement type
    case 'line',                                                    % Linear movement
        center      = object.center + t*move.direction;
    case 'circ',                                                    % Circular movement
        tt          = move.t0 + t*move.dtheta;
        center      = floor(move.center + move.radius*[cos(tt),sin(tt)]);
    case 'expd',
        center      = object.center;
    end

    %%%%% Convert center into cells of the current level
    for l = o+1:k                                                   % If we're at a finer grid, convert coordinates
        center      = convert_c2f(l-1,center,'cell');               % Convert cell coordinates of Level l-1 to l
    end
        
    %%%%% Prepare a list of cells that consitute the object, around its center
    switch (object.type)                                            % Prepare the object
    case 'box',                                                     % Create a box        
        s           = box_list(floor(center - object.size/2),floor(center + object.size/2));
    case 'ball',                                                    % Create a ball (disc)
        if (move.type == 'expd')            
            radius      = t*move.dradius;                            % Convert ball radius to current level cell coordinates
        else
            radius      = object.radius;
        end
        for l = o+1:k                                               % If we're at a finer grid, convert coordinates
            radius  = convert_c2f(l-1,radius,'cell');               % Convert cell coordinates of Level l-1 to l
        end

        switch (k)                                                  % Do different things at different levels
        case o,            
            s           = ball_list(center,radius);                 % k=o: full ball
            if (move.type == 'expd')                               % Use an annulus if this is the expanding ball case
                s_inner     = ball_list(center,floor(radius/2));      % Inner ball to be removed
                s           = setdiff(s,s_inner,'rows');                % Remove inner ball from outer ball
            end
        case o+1,
            s           = ball_list(center,radius);                 % k=o+1: an annulus inside the ball of k=o
            s_inner     = ball_list(center,floor(3*radius/4));      % Inner ball to be removed
            s           = setdiff(s,s_inner,'rows');                % Remove inner ball from outer ball
        case o+2,
            s           = ball_list(center,radius);                 % k=o+2: half an annulus inside the ball of k=o
            s_inner     = ball_list(center,floor(3*radius/4));      % Inner ball to be removed
            s           = setdiff(s,s_inner,'rows');                % Remove inner ball from outer ball
            left        = find(s(:,1) < center(1));
            s(left,:)   = [];                                       % Delete the left half => half an annulus
        end
    end       

    inner           = [];                                           % Remove points that are outside the domain from a
    for p = o:size(L{k}.patch_active,1)+o-1,
        j           = L{k}.patch_active(p,:);                       % d-D coordinates of the patch
        inner       = union(inner,find(check_range(s,patch_start(k,j),patch_finish(k,j))));
    end
   
    points          = [points; s(inner,:)];                         % Add flagged cells of this object to the global list
end
