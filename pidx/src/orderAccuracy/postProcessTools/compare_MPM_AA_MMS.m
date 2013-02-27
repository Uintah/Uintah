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
  printf('  -norm  (Linf or L2) - prints Linf the output file \n')
  printf('  -MMS  (1 or 2)      - 1:  1D Periodic Bar MMS \n')
  printf('                      - 2:  3D Axis-Aligned  MMS \n')
  printf('  -o <fname>          - Dump the output (LInfError) to a file\n')           
end 

%__________________________________
% defaults
ts          = 999;
output_file = 'L_inf';
whichNorm   = 'L2';
MMS         = 999;

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
  elseif (strcmp(option,"-norm") )  
    whichNorm = opt_value;
  elseif (strcmp(option,"-MMS") )  
    MMS = str2num(opt_value);
  end         
end

if( strcmp(whichNorm,"L2")==0 && strcmp(whichNorm,"Linf")== 0)
  disp( 'compare_MPM_AA_MMS.m: invalid norm option');
  Usage
  disp( 'Now exiting....');
  exit(-1)
end

if( MMS != 1 && MMS != 2)
  disp( 'compare_MPM_AA_MMS.m: invalid (MMS) option');
  Usage
  disp( 'Now exiting....');
  exit(-1)
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
c1 = sprintf('puda -timesteplow %i -AA_MMS_%i %s  >& tmp',ts-1,MMS,uda);
[s1, r1] = unix(c1);

data = load('L_norms'); 

norm = -9;
if( strcmp(whichNorm, "Linf") )
  norm = data(2);
elseif(  strcmp(whichNorm, "L2") )
  norm = data(3);
end


% write data to the output file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',norm);
  fclose(fid);
end


