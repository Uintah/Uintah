%function compareMatrices
%COMPAREMATRICES  Compare the Uintah and Hypre solver matrices.
%   This script compares the two matrices used for the pressure solve
%   inside implicit ICE:
%   (1) The matrix constructed in Uintah (7-point stencil on a 3D uniform
%   grid). Used directly in Steve's solver.
%   (2) The matrix input to the Hypre solvers, in Hypre format.
%   This script loads both matrices to MATLAB and compares their entries to
%   look for possible discrepencies.
%   See also: LOAD.

% Revision history:
% 05-JUL-2005   Oren      Created

%=================================================================
% Load Uintah matrix file
% Format of every row: i1 i2 i3 A.b A.w A.s A.p A.n A.e A.t
%=================================================================

dim                     = 3;

offset                  = 2;
uintah                  = load('uintah.A.matrix','-ascii');
subList                 = uintah(:,1:3);
numCells                = max(subList,[],1) - min(subList,[],1) + 1;

% Remove extra cell dummy equations 
extraCells              = [];
for d = 1:dim
    extraCells = union(extraCells,...
        find((subList(:,d) < 2-offset) | (subList(:,d) > numCells(d)-1-offset)));       
end
uintah(extraCells,:)    = [];
subList(extraCells,:)   = [];

% Move to 1-based subscripts and translate to cell indices
offset                  = 1;
numCells                = numCells-2;

subCell                 = cell(dim,1);
for d = 1:dim
    subCell{d}          = subList(:,d) + offset;
end
indCell                 = sub2ind(numCells,subCell{:});

offsets = [...
    [0 0 -1]; ...
    [-1 0 0]; ...
    [0 -1 0]; ...
    [0 0 0]; ...
    [0 1 0]; ...
    [1 0 0]; ...
    [0 0 1]; ...
    ];

%=================================================================
% Load Hypre matrix file
% Format of every row: i1 i2 i3 A.w A.s A.p A.n A.e A.t
%=================================================================
