#! /usr/bin/octave -qf
%_________________________________
%
%  Example usage:
%  gradient.m -dat -pDir
%_________________________________
clear all;
close all;
format long e;

function Usage
  printf('gradient.m <options>\n')
  printf('options:\n')
  printf('  -dat  name of the dat file \n')
  printf('  -pDir principal direction to take the gradient \n')
end

%________________________________
% Usage
nargin = length(argv);
if (nargin == 0)
  Usage
  exit
endif

%__________________________________
% read in inputs
arg_list = argv ();
for i = 1:2:nargin
   option    = sprintf("%s",arg_list{i} );
   opt_value = sprintf("%s",arg_list{++i});

  if ( strcmp(option,"-dat") )
    dat  = opt_value;
  elseif (strcmp(option,"-pDir") ) 
    pDir = str2num(opt_value);
  end
end

% remove [] from file
c = sprintf( 'cat %s | tr -d \"[]\" >dat; mv dat %s', dat,dat);
unix(c);
  
% load array
data = load(dat);
NxM = size(data);

%__________________________________
% which column contains the data
if (NxM(2) == 2)   % time series data 
  nCol = 2;  
end

if (NxM(2) == 4)   % Scalar data [x, y, z, Q]
  nCol = 4;   
end

if (NxM(2) > 4)    % Vector Data [x, y, z, Q.x, Q.y, Q.z]
  nCol = pDir + 3;
end

% compute the gradient
var  = data(:,nCol);
x    = data(:,pDir);
dx   = x(2) - x(1);
gradVar = gradient(var, dx);

% add gradient to matrix
newData = horzcat(data,gradVar);

% print the new matrix
NxM = size(newData);

fid = fopen (dat, "w");

for i = 1:NxM(1)
  for j = 1:NxM(2)
    fprintf(fid, '%16.15E  ',newData(i,j));
  end
  fprintf(fid,'\n');
end



