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
% default user inputs;69
ts          = 999;
output_file = 'LnError';

% parse the inputs
arg_list = argv ();
for i = 1:2:nargin
   option    = sprintf("%s",arg_list{i} );
   opt_value = sprintf("%s",arg_list{++i});

  if ( strcmp(option,"-uda") )   
    uda = opt_value;
  elseif (strcmp(option,"-ts") )
    ts = str2num(opt_value);                  
  elseif (strcmp(option,"-o") )  
    output_file = opt_value;    
  end                                      
end

%________________________________
% determine which files to read in.
x_fname_error = 'totalummsLnError.dat';
y_fname_error = 'totalvmmsLnError.dat';

x_fname_exact = 'totalummsExactSol.dat';
y_fname_exact = 'totalvmmsExactSol.dat';


% import the data
x_c0 = sprintf('%s/%s',uda,x_fname_error);
y_c0 = sprintf('%s/%s',uda,y_fname_error);

x_c1 = sprintf('%s/%s',uda,x_fname_exact);
y_c1 = sprintf('%s/%s',uda,y_fname_exact);

x_error  = load(x_c0); 
y_error  = load(y_c0); 

x_exact  = load(x_c1);
y_exact  = load(y_c1);

last = min(ts,length(x_error(:,2)));

x_LnError = sqrt(x_error(:,2))/sqrt(x_exact(:,2))
y_LnError = sqrt(y_error(:,2))/sqrt(y_exact(:,2))

% write LnError to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g %g\n',x_LnError(last), y_LnError(last));
  fclose(fid);
end
