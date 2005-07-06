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
fprintf('Loading Uintah matrix\n');
% We assume the file was stripped from any character except the
% aforementioned columns of data (ints/floats).

dim                     = 3;        % Problem dimension

uintah                  = load('uintah.A.matrix','-ascii');
uList                   = zeros(0,3);
bigOffset               = 2;
listCell                = uintah(:,1:3);
bigNumCells             = max(listCell,[],1) - min(listCell,[],1) + 1;

% Remove extra cell dummy equations 
extraCells              = [];
for d = 1:dim
    extraCells = union(extraCells,...
        find((listCell(:,d) < 2-bigOffset) | ...
        (listCell(:,d) > bigNumCells(d)-1-bigOffset)));       
end
uintah(extraCells,:)    = [];
listCell(extraCells,:)  = [];

% Move to 1-based subscripts and translate to cell indices
offset                  = 1;
numCells                = bigNumCells-2;
numCellsTotal           = prod(numCells);
subCell                 = cell(dim,1);
for d = 1:dim
    subCell{d}          = listCell(:,d) + offset;
end
indCell                 = sub2ind(numCells,subCell{:});

% Corresponding to the column ordering of stencil entries
stencilOffsets = [...   
    [0 0 -1]; ...                   % A.b (z-left)
    [-1 0 0]; ...                   % A.w (x-left)
    [0 -1 0]; ...                   % A.s (y-left)
    [0 0 0]; ...                    % A.p (diagonal coefficient)
    [0 1 0]; ...                    % A.n (y-right)
    [1 0 0]; ...                    % A.e (x-right)
    [0 0 1]; ...                    % A.t (z-right)
    ];

for j = 1:size(stencilOffsets,1)
    col = j+3;
    fprintf('Adding data for stencil entry (%+d,%+d,%+d), file column %2d\n',...
        stencilOffsets(j,:),col);
    listNbhr            = listCell + ...
        repmat(stencilOffsets(j,:),size(listCell)./size(stencilOffsets(j,:)));

    % Remove extra cell variables from neighbour list
    extraNbhrs          = [];
    for d = 1:dim
        extraNbhrs = union(extraNbhrs,...
            find((listNbhr(:,d) < 2-bigOffset) | ...
            (listNbhr(:,d) > bigNumCells(d)-1-bigOffset)));
    end
    listNbhr(extraNbhrs,:) = 0;                 % Dummy values that won't crash the sub2ind call below

    % Move to 1-based subscripts and translate to nbhr cell indices
    subNbhr             = cell(dim,1);
    for d = 1:dim
        subNbhr{d}      = listNbhr(:,d) + offset;
    end
    indNbhr             = sub2ind(numCells,subNbhr{:});

    % Add only relevant list rows to list of non-zeros
    legal   = setdiff(1:numCellsTotal,extraNbhrs);
    uList   = [uList; [indCell(legal) indNbhr(legal) uintah(legal,col)]];
end

%=================================================================
% Load Hypre matrix file
% Format of every row: i1 i2 i3 A.w A.s A.p A.n A.e A.t
%=================================================================









