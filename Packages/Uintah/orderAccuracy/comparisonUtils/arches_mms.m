#! /usr/bin/octave -qf
%_________________________________
%
%  Example usage:
%  arches_mms.m -pDir 1 -o out.400.cmp -uda almgren_MMS_128.uda
%_________________________________
clear all;
close all;
format short e;

function Usage
  printf('arches_mms.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n')                                                  
  printf('  -pDir <1,2,3>       - principal direction \n')                                                  
  printf('  -ts                 - Timestep to compute L2 error, default is the last timestep\n') 
  printf('  -o <fname>          - Dump the output (LnError) to a file\n')                                    
end 

%________________________________            
% dump out usage
nargin = length(argv);
if (nargin == 0)
  Usage
  exit
endif

%__________________________________
% default user inputs
pDir       = 1;
ts          = 999;
output_file = 'LnError';

% parse the inputs
arg_list = argv ();
for i = 1:2:nargin
   option    = sprintf("%s",arg_list{i} );
   opt_value = sprintf("%s",arg_list{++i});

  if ( strcmp(option,"-uda") )   
    uda = opt_value;
  elseif (strcmp(option,"-pDir") ) 
    pDir = str2num(opt_value);
  elseif (strcmp(option,"-ts") )
    ts = str2num(opt_value);                  
  elseif (strcmp(option,"-o") )  
    output_file = opt_value;    
  end                                      
end

%________________________________
% determine which files to read in.
if(pDir == 1)
  fname_error = 'totalummsLnError.dat';
  fname_exact = 'totalummsExactSol.dat';
elseif(pDir == 2)
  fname_error = 'totalvmmsLnError.dat';
  fname_exact = 'totalvmmsExactSol.dat';
elseif(pDir == 3)
  % to be filled in later
end

% import the data
c0 = sprintf('%s/%s',uda,fname_error);
c1 = sprintf('%s/%s',uda,fname_exact);
error  = load(c0); 
exact  = load(c1);

last = min(ts,length(error(:,2)));

LnError = error(:,2)/exact(:,2)

% write LnError to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',LnError(last));
  fclose(fid);
end
