%______________________________________________________________________
% This octave function tests for the existance of puda, lineextract , then extracts physical
% time and grid info.
%______________________________________________________________________

function [TG] = getTimeGridInfo (uda, ts, level)
  
  % bulletproofing
  % do the Uintah utilities exist
  
  [s0, r0] = unix('puda > /dev/null 2>&1');
  [s1, r1] = unix('lineextract > /dev/null 2>&1');
  [s2, r2] = unix('timeextract > /dev/null 2>&1');

  flag = 0;
  if( s0 ~=1 || s1 ~= 1 || s2 ~= 1)
    disp('Cannot find one of the uintah utilities (puda, lineextract, timeextract)');
    disp('  a) make sure you are in the right directory, and');
    disp('  b) the utilities have been compiled');
    quit(-1);
    flag=1;
  end

  %________________________________
  %  extract the physical time
  c = sprintf('puda -timesteps %s | grep ''^[0-9]'' | awk ''{print $2}'' > tmp 2>&1',uda);
  [s, r]=unix(c);
  physicalTime  = load('tmp');
  
  if(ts == 999)  % default
    ts = length(physicalTime);
  endif

  TG.ts = int8(ts);
  TG.physicalTime = physicalTime(ts);

  %________________________________
  %  extract the grid information on a level
  c = sprintf('puda -gridstats %s > tmp 2>&1',uda); 
  [s,r] = unix(c);
  
  c1 = sprintf('sed -n /"Level: index %i"/,/"dx"/{p} tmp > tmp.clean 2>&1',level);
  [s,r] = unix(c1);
  
  [s,r0] = unix('grep -m1 dx: tmp.clean                  | tr -d "dx:[]"');
  [s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp.clean |cut -d":" -f2 | tr -d "[]int"');
  [s,r2] = unix('grep -m1 -w "Domain Length" tmp         |cut -d":" -f2 | tr -d "[]"');

  TG.dx = str2num(r0);
  TG.resolution   = int64( str2num(r1) );
  TG.domainLength = str2num(r2);
  
  %cleanup
  [s,r] = unix('/bin/rm tmp tmp.clean');
endfunction
