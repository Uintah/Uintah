function [grid,A,b,T,TI] = setupGrid
%SETUPGRID  Set up the grid hierarchy.
%   GRID = SETUPGRID sets up the AMR grid hierarchy GRID, based on the
%   parameters of param and several hard-coded ("static") refinement
%   techniques in this function.
%
%   See also: TESTDISC, TESTADAPTIVE.

% Revision history:
% 16-JUL-2005    Oren Livne    Created

globalParams;

distrib = 0; % Make that 1 to match distributed patches to multiple procs in C++


out(2,'--- setupGrid() BEGIN ---\n');
tStartCPU       = cputime;
tStartElapsed   = clock;

A                   = [];
b                   = [];
T                   = [];
TI                  = [];
grid                = [];

grid.dim            = param.dim;
grid.domainSize     = param.domainSize;
grid.maxLevels  	= param.maxLevels;
grid.maxPatches  	= param.maxPatches;
grid.numLevels      = 0;
grid.totalVars      = 0;
grid.level          = cell(grid.maxLevels,1);

%--------------- Level 1: global coarse grid -----------------

resolution          = repmat(param.baseResolution,[1 grid.dim]);
[grid,k]            = addGridLevel(grid,'meshsize',grid.domainSize./resolution);

if (~distrib)
    % 1 Patch
    [grid,q1]           = addGridPatch(grid,k,ones(1,grid.dim),resolution,-1);     % One global patch
else
    % 2^d Patches over same domain
    offset              = fliplr(graycode(grid.dim,repmat(2,[grid.dim 1])));
    q1                  = zeros(size(offset,1),1);
    for i = 1:size(offset,1)
        ilower  = ones(1,grid.dim) + offset(i,:).*resolution/2;
        iupper  = ilower + resolution/2 - 1;
        [grid,q1(i)]     = addGridPatch(grid,k,ilower,iupper,-1);
    end
end

for q = 1:grid.level{k}.numPatches,
    [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
end

%--------------- Level 2: local fine grid around center of domain -----------------

if (param.twoLevel)
    [grid,k]            = addGridLevel(grid,'refineRatio',repmat(2,[1 grid.dim]));
    switch (param.twoLevelType)
        case 'global',
            % Cover the entire domain
            [grid,q2]  = addGridPatch(grid,k,ones(1,grid.dim),2*resolution,q1);              % Local patch around the domain center
        case 'centralHalf',

            if (~distrib)
                % Cover central half of the domain
                [grid,q2]  = addGridPatch(grid,k,resolution/2 + 1,3*resolution/2,q1);              % Local patch around the domain center
            else
                % 2^d Patches over same domain
                offset              = fliplr(graycode(grid.dim,repmat(2,[grid.dim 1])));
                q2                  = zeros(size(offset,1),1);
                for i = 1:size(offset,1)
                    ilower  = resolution/2 + 1 + offset(i,:).*resolution/2;
                    iupper  = ilower + resolution/2 - 1;
                    [grid,q2(i)]     = addGridPatch(grid,k,ilower,iupper,q1(i));
%                    [grid,q2(i)]     = addGridPatch(grid,k,ilower,iupper,q1);
                end
            end
            
            
        case 'centralQuarter',
            % Cover central quarter of the domain
            [grid,q2]  = addGridPatch(grid,k,3*resolution/4 + 1,5*resolution/4,q1);              % Local patch around the domain center
        case 'leftHalf',
            % Cover left half of the domain in x1
            ilower      = ones(size(resolution));
            iupper      = 2*resolution;
            iupper(1)   = resolution(1);
            [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
        case 'rightHalf',
            % Cover right half of the domain in x1
            ilower      = ones(size(resolution));
            ilower(1)   = resolution(1) + 1;
            iupper      = 2*resolution;
            [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
        case 'nearXMinus',
            % A patch next to x-minus boundary, covers the central
            % half of it, and extends to half of the domain in x.
            ilower      	= ones(size(resolution));
            ilower(2:end)	= resolution(2:end)/2 + 1;
            iupper          = ilower + resolution - 1;
            iupper(1)       = ilower(1) + resolution(1)/2 - 1;
            [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
        case 'centralHalf2Patches',
            % Two fine patches next to each other at the center of the
            % domain
            ilower = resolution/2 + 1;
            iupper = 3*resolution/2;
            iupper(1) = resolution(1);
            [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
            ilower = resolution/2 + 1;
            iupper = 3*resolution/2;
            ilower(1) = resolution(1)+1;
            [grid,q3]  = addGridPatch(grid,k,ilower,iupper,q1);
        case 'centralQuarter2Patches',
            % Two fine patches next to each other at the central
            % quarter of the domain
            ilower = 3*resolution/4 + 1;
            iupper = 5*resolution/4;
            iupper(1) = resolution(1);
            [grid,q2]  = addGridPatch(grid,k,ilower,iupper,q1);
            ilower = 3*resolution/4 + 1;
            iupper = 5*resolution/4;
            ilower(1) = resolution(1)+1;
            [grid,q3]  = addGridPatch(grid,k,ilower,iupper,q1);
        otherwise,
            error('Unknown two level type');
    end

    for q = 1:grid.level{k}.numPatches,
        [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
    end
end

%--------------- Level 3: yet local fine grid around center of domain -----------------
if ((param.twoLevel) & (param.threeLevel))
    [grid,k]   = addGridLevel(grid,'refineRatio',[2 2]);
    switch (param.threeLevelType)
        case 'centralHalf',
            % Cover central half of the domain
            [grid,q3]  = addGridPatch(grid,k,3*resolution/2 + 1,5*resolution/2,q2);              % Local patch around the domain center
        case 'centralHalfOfcentralQuarter',
            % Cover central half of the central quarter of the domain
            [grid,q3]  = addGridPatch(grid,k,7*resolution/4 + 1,9*resolution/4,q2);              % Local patch around the domain center
            %                [grid,q3]  = addGridPatch(grid,k,15*resolution/4 + 1,17*resolution/4,q2);              % Local patch around the domain center
        otherwise,
            error('Unknown three level type');
    end

    for q = 1:grid.level{k}.numPatches,
        [grid,A,b,T,TI]      = updateSystem(grid,k,q,A,b,T,TI);
    end
end

if (param.verboseLevel >= 2)
    printGrid(grid);
end
tCPU            = cputime - tStartCPU;
tElapsed        = etime(clock,tStartElapsed);
out(2,'CPU time     = %f\n',tCPU);
out(2,'Elapsed time = %f\n',tElapsed);
out(2,'--- setupGrid() END ---\n');
