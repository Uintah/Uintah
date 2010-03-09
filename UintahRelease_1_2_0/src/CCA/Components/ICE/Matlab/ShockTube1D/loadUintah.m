%LOADUINTAH  Load solution data from Uintah output for the Shock Tube
%   Problem in 1D.
%
%   Here we load the state variables of the Euler equations as a reference
%   for our matlab simulation of Uintah's ice, ICE.M. This is a script, not a
%   function.
%
%   See also ICE, PLOTRESULTS.

%cd /scratch/SCIRun_Fresh/linux32dbg/Packages/Uintah/StandAlone;
uda = '/data/Builds/Fresh/111109SCIRun/dbg_Linux/StandAlone/test1.uda';
delX = 0.01;

%  extract the physical time for each dump
!rm -f tmp
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
fprintf('Uintah output program puda command string: ''%s''\n',c0);
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
if (tstep > 1)
    fprintf('Uintah delT = %e   ',physicalTime(tstep) - physicalTime(tstep -1));
    fprintf('MATLAB delT = %e   ',delT);
    fprintf('rel.diff = %e\n', abs(physicalTime(tstep) - physicalTime(tstep -1)-delT)/abs(delT));
end

ts = tstep -1;
startEnd = '-istart -1 0 0 -iend 1000 0 0';
unix( '/bin/rm -f rho vel_tmp temp press delP press_eq uvelFC');
c1 = sprintf('lineextract -v rho_CC         -cellCoords -timestep %i %s -o rho      -m 0 -uda %s',ts,startEnd,uda);
c2 = sprintf('lineextract -v vel_CC         -cellCoords -timestep %i %s -o vel_tmp  -m 0 -uda %s',ts,startEnd,uda);
c3 = sprintf('lineextract -v temp_CC        -cellCoords -timestep %i %s -o temp     -m 0 -uda %s',ts,startEnd,uda);
c4 = sprintf('lineextract -v press_CC       -cellCoords -timestep %i %s -o press    -m 0 -uda %s',ts,startEnd,uda);
c5 = sprintf('lineextract -v delP_Dilatate  -cellCoords -timestep %i %s -o delP     -m 0 -uda %s',ts,startEnd,uda);
c6 = sprintf('lineextract -v press_equil_CC -cellCoords -timestep %i %s -o press_eq -m 0 -uda %s',ts,startEnd,uda);
c7 = sprintf('lineextract -v uvel_FCME      -cellCoords -timestep %i %s -o uvelFC   -m 0 -uda %s',ts,startEnd,uda);

[status1, result1]=unix(c1);
[status2, result2]=unix(c2);
[status3, result3]=unix(c3);
[status4, result4]=unix(c4);
[status5, result5]=unix(c5);
[status6, result6]=unix(c6);
[status7, result7]=unix(c7);

% rip out [] from velocity data
c8 = sprintf('sed ''s/\\[//g'' vel_tmp | sed ''s/\\]//g'' >vel');
[status8, result8]=unix(c8);

delP_ice      = importdata('delP');
press_ice     = importdata('press');
temp_ice      = importdata('temp');
rho_ice       = importdata('rho');
vel_ice       = importdata('vel');
uvel_FC_ice   = importdata('uvelFC');
press_eq_ice  = importdata('press_eq');
uvel_FC_ice   = importdata('uvelFC');
x_ice         = press_ice(:,1);
x_FC_ice      = uvel_FC_ice(:,1);

Ncells     = length(x_ice);
N_FC_cells = length(x_FC_ice);

x_ice         = reshape(x_ice,              1, Ncells);
x_FC_ice      = reshape(x_FC_ice,           1, N_FC_cells);
rho_ice       = reshape(rho_ice(:,4) ,      1, Ncells);
vel_ice       = reshape(vel_ice(:,4) ,      1, Ncells);
temp_ice      = reshape(temp_ice(:,4) ,     1, Ncells);
press_ice     = reshape(press_ice(:,4) ,    1, Ncells);
delP_ice      = reshape(delP_ice(:,4) ,     1, Ncells);
press_eq_ice  = reshape(press_eq_ice(:,4) , 1, Ncells);
uvel_FC_ice    = reshape(uvel_FC_ice(:,4) , 1, N_FC_cells);

unix('/bin/rm delP press temp rho vel uvelFC vel_tmp');
