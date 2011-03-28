#! /usr/bin/octave -qf
%_________________________________
% This octave file is a wrapper for 
% puda -AA_MMS
%
%  Example usage:
%  compare_MPM_AA_MMS.m  -o out.16.cmp -uda AA_MMS.uda
%_________________________________
clear all;
close all;
format short e;

function Usage
  printf('compare_MPM_AA_MMS.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n')                                                  
  printf('  -ts                 - Timestep to compute L-inf error, default is the last timestep\n') 
  printf('  -o <fname>          - Dump the output (LInfError) to a file\n')           
end 

%__________________________________
% defaults
ts          = 999;
output_file = 'L_inf';
L           = 0;

%________________________________            
% Parse User inputs  
%echo
nargin = length(argv);
if (nargin == 0)
  Usage
  exit
endif

% Parse the command line arguments
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
% do the Uintah utilities exist
[s0, r0]=unix('puda >& /dev/null');

if( s0 ~=0  )
  disp('Cannot execute uintah utilites puda');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilitie (puda) have been compiled');
end

%________________________________
%  extract the physical time
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime  = load('tmp');

if(ts == 999)  % default
  ts = length(physicalTime)
endif

%__________________________________
% compute error
c1 = sprintf('puda -timesteplow %i -AA_MMS %s  >& tmp',ts-1,uda);
[s1, r1] = unix(c1);
L_inf = load('L_inf'); 

% write L_inf to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',L_inf);
  fclose(fid);
end


