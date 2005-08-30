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
% Format of every row: "[int" i1 i2 i3 "] A.b" A.b " A.w " ...
% A.w " A.s " A.s " A.p " A.p " A.n " A.n " A.e " A.e " A.t " A.t
%=================================================================
fprintf('Loading Uintah matrix\n');
% We assume the file was stripped from any headers/tails so that it
% contains only the matrix data.

dim                     = 3;        % Problem dimension

%uintah                  = load('uintah.A.matrix','-ascii');
f = fopen('uintah.A.matrix','r');
uintah = fscanf(f,'[int %d, %d, %d] A.b %f A.w %f A.s %f A.p %f A.n %f A.e %f A.t %f\n',...
    inf);
fclose(f);
uintah                  = reshape(uintah,[10 length(uintah)/10])';
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
numStencilEntries = size(stencilOffsets,1);

for j = 1:numStencilEntries,
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

U = spconvert(uList);               % Convert non-zeros list to sparse matrix

%=================================================================
% Load Uintah vector file
% Format of every row: "[int" i1 i2 i3 "] A.b" A.b " A.w " ...
% A.w " A.s " A.s " A.p " A.p " A.n " A.n " A.e " A.e " A.t " A.t
%=================================================================
fprintf('Loading Uintah vector\n');
% We assume the file was stripped from any headers/tails so that it
% contains only the matrix data.

dim                     = 3;        % Problem dimension

%uintah                  = load('uintah.A.matrix','-ascii');
f = fopen('uintah.B.vector','r');
uintah = fscanf(f,'[%d, %d, %d]~ %f\n',inf);
fclose(f);
uintah                  = reshape(uintah,[4 length(uintah)/4])';
bigOffset               = 2;
listCell                = uintah(:,1:3);
bigNumCells             = max(listCell,[],1) - min(listCell,[],1) + 3;

% Remove extra cell dummy equations
extraCells              = [];
for d = 1:dim
    extraCells = union(extraCells,...
        find((listCell(:,d) < 2-bigOffset) | ...
        (listCell(:,d) > bigNumCells(d)-1-bigOffset)));
end
uintah(extraCells,:)    = [];
listCell(extraCells,:)  = [];
UB = uintah(:,4);

%=================================================================
% Load Hypre matrix file
% Format of every row: i1 i2 i3 A.w A.s A.p A.n A.e A.t
%=================================================================
fprintf('\nLoading Hypre matrix\n');
f                       = fopen('hypre.HA.00000','r');
interface               = fscanf(f,'%s\n\n',1);
symmetric               = fscanf(f,'Symmetric: %d\n\n',1);
constcoef               = fscanf(f,'ConstantCoefficient: %d\n\n',1);
dim                     = fscanf(f,'Grid:\n%d\n',1);
numParts                = fscanf(f,'%d\n',1);
parts                   = fscanf(f,'%d: (%d, %d, %d)  x  (%d, %d, %d)\n',[numParts 7]);
part                    = parts(:,1);
lower                   = parts(:,2:4);
upper                   = parts(:,5:7);
numCells                = upper-lower+1;
numStencilEntries       = fscanf(f,'\nStencil:\n%d\n',1);
stencil                 = fscanf(f,'%d: %d %d %d\n',[4 numStencilEntries])';
stencilEntry            = stencil(:,1);
stencilOffsets          = stencil(:,2:dim+1);
temp                    = fscanf(f,'\n%s:\n',1);      % temp = 'Data'
hypre                   = fscanf(f,'%d: (%d, %d, %d; %d) %f\n',inf);
fclose(f);

hypre                   = reshape(hypre,[6 length(hypre)/6])';
hList                   = zeros(0,3);
listCell                = hypre(:,2:4);
listNbhr                = listCell + stencilOffsets(hypre(:,5)+1,:);

% Remove extra cell dummy equations
extraCells              = [];
for d = 1:dim
    extraCells = union(extraCells,...
        find((listCell(:,d) < lower(d)) | ...
        (listCell(:,d) > upper(d))));
end
for d = 1:dim
    extraCells = union(extraCells,...
        find((listNbhr(:,d) < lower(d)) | ...
        (listNbhr(:,d) > upper(d))));
end
hypre(extraCells,:)     = [];
listCell(extraCells,:)  = [];
listNbhr(extraCells,:)  = [];

% Move to 1-based subscripts and translate to cell indices
offset                  = 1;
numCellsTotal           = prod(numCells);
subCell                 = cell(dim,1);
for d = 1:dim
    subCell{d}          = listCell(:,d) + offset;
end
indCell                 = sub2ind(numCells,subCell{:});

subNbhr                 = cell(dim,1);
for d = 1:dim
    subNbhr{d}          = listNbhr(:,d) + offset;
end
indNbhr                 = sub2ind(numCells,subNbhr{:});

% Add rows to list of non-zeros
hList                   = [hList; [indCell indNbhr hypre(:,6)]];
H                       = spconvert(hList);

%=================================================================
% Load Hypre RHS vector file
%=================================================================
fprintf('\nLoading Hypre vector\n');
f                       = fopen('hypre.HB.00000','r');
interface               = fscanf(f,'%s\n\n',1);
dim                     = fscanf(f,'Grid:\n%d\n',1);
numParts                = fscanf(f,'%d\n',1);
parts                   = fscanf(f,'%d: (%d, %d, %d)  x  (%d, %d, %d)\n',[numParts 7]);
part                    = parts(:,1);
lower                   = parts(:,2:4);
upper                   = parts(:,5:7);
numCells                = upper-lower+1;
temp                    = fscanf(f,'\n%s:\n',1);      % temp = 'Data'
hypre                   = fscanf(f,'%d: (%d, %d, %d; %d) %f\n',inf);
fclose(f);

hypre                   = reshape(hypre,[6 length(hypre)/6])';
listCell                = hypre(:,2:4);

% Remove extra cell dummy equations
extraCells              = [];
for d = 1:dim
    extraCells = union(extraCells,...
        find((listCell(:,d) < lower(d)) | ...
        (listCell(:,d) > upper(d))));
end
hypre(extraCells,:)     = [];
listCell(extraCells,:)  = [];

% Move to 1-based subscripts and translate to cell indices
offset                  = 1;
numCellsTotal           = prod(numCells);
subCell                 = cell(dim,1);
for d = 1:dim
    subCell{d}          = listCell(:,d) + offset;
end
indCell                 = sub2ind(numCells,subCell{:});

% Add rows to list of non-zeros
HB                     = hypre(:,6);

%=================================================================
% Measure discrepancy between U and H
%=================================================================
fprintf('\n');
different               = length(find(abs(H-U) > 1e-10));
fprintf('# different entries in Uintah/Hypre = %d / %d total\n',...
    different,length(find(U)));
