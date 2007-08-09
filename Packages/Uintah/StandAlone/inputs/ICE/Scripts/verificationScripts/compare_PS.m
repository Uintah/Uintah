#! /usr/bin/octave -qf

%_________________________________
% 04/22/07   --Amjith (ramanuja@cs.utah.edu)
% This matLab script computes the err diff between 
% exact sol and uda file for the 1-D advection of 
  % passive scalar
  %
  %  NOTE:  You must have 
  %  setenv MATLAB_SHELL "/bin/tcsh -f"
  %  in your .cshrc file for this to work
    %_________________________________

    close all;
    clear all;
    clear function;
      

    function Usage
    printf('compare_PS.m <options>\n')
    printf('options:\n')
    printf('  -uda  <udaFileName> - name of the uda file \n')
    printf('  -type  <type>       - type of passive scalar [linear|quad|cubic|exp|sine]\n')
    printf('  -vel   <num>        - Velocity of advection in m/s\n')
    printf('  -min   <num>        - minimum X value of the exact solution\n')
    printf('  -max   <num>        - maximum X value of the exact solution\n')
    printf('  -cells <\'string\'>   - This cells option is an input to the lineextract eg: \'-istart 0 0 0 -iend 99 0 0\'\n')
    printf('  -freq  <num>        - Frequency used by the sine profile\n')
    printf('  -coeff <num>        - coeff used by the quad or exp profile\n')
    printf('  -slope <num>        - slope of the linear profile\n')
    printf('  -L                  - Compute L2 error for last timestep only (useful for testing framework)\n')
    printf('  -o <fname>          - Dump the output (L2Error) to a file\n')
    end 
%      argv = {"-uda";"advectCubic__r100v10.uda.001";"-type"; "cubic";"-vel";"10";"-min";"-0.1";"-max";"0.1";"-cells";"-istart 0 0 0 -iend 99 0 0";"-coeff";"10";"-L";"-o";"test_out.txt"}

 %     nargin = length(argv)


      if (nargin == 0)
         Usage
	 exit
      endif 

      for i = 1:nargin
	
	%________________________________
	% USER INPUTS
	
	
	if strcmp(sprintf("%s",argv(i,:)),"-uda")
	  uda = sprintf("%s",argv(++i,:))
	elseif strcmp(sprintf("%s",argv(i,:)),"-type")
	  exactSolution = sprintf("%s",argv(++i,:)) %'exp' %'linear' %'sinusoidal'; %linear; %cubic; %quad
	elseif strcmp(sprintf("%s",argv(i,:)),"-vel")
	  velocity  = str2num(sprintf("%s",argv(++i,:)))
	elseif strcmp(sprintf("%s",argv(i,:)),"-min")
	  exactSolMin =  str2num(sprintf("%s",argv(++i,:)))
	elseif strcmp(sprintf("%s",argv(i,:)),"-max")
	  exactSolMax =  str2num(sprintf("%s",argv(++i,:)))
	elseif strcmp(sprintf("%s",argv(i,:)),"-cells")
	  startEnd =  sprintf("%s",argv(++i,:))
	elseif strcmp(sprintf("%s",argv(i,:)),"-freq")
	  freq =  str2num(sprintf("%s",argv(++i,:)))
	elseif strcmp(sprintf("%s",argv(i,:)),"-slope")
	  slope =  str2num(sprintf("%s",argv(++i,:)))
	elseif strcmp(sprintf("%s",argv(i,:)),"-coeff")
	  coeff =  str2num(sprintf("%s",argv(++i,:))) 
	elseif strcmp(sprintf("%s",argv(i,:)),"-L")
	  last = true
	elseif strcmp(sprintf("%s",argv(i,:)),"-o")
	  output_file = sprintf("%s",argv(++i,:))
	endif

      endfor

      fid = fopen(output_file, 'w');

      %________________________________
      %  extract the physical time
      c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda)
      [status0, result0]=unix(c0);
      physicalTime  = load('tmp');
      nDumps = length(physicalTime)

      if (last)
	startVal = nDumps;
      end


      %_________________________________
      % Loop over all the timesteps
      for(n = startVal:nDumps )
	%  n = input('input timestep') 
	ts = n-1;
	
	time = sprintf('%e sec',physicalTime(n));
	
	
	S_E = startEnd;
	unix('/bin/rm -f scalar-f.dat');


	
	%____________________________
	%   scalar-F
        c6 = sprintf('lineextract -v scalar-f -l 0 -cellCoords -timestep %i %s -o scalar-f.dat -m 0  -uda %s',ts,S_E,uda)
        [s6, r6]=unix(c6);
        scalarArray = load('scalar-f.dat');
        xx     = scalarArray(:,1);
        scalar = scalarArray(:,4);


        %_________________________________
        % Exact Solution on each level
        dist   = exactSolMax - exactSolMin;
        offset = physicalTime(n) * velocity;
        xmin   = exactSolMin + offset;
        xmax   = exactSolMax + offset;
        uda_dx = xx(2) - xx(1);
        x = xmin:uda_dx:xmax;
        exactSol=xx * 0;
        
        length(xx)

        for( i = 1:length(xx))
          if(xx(i) >= xmin && xx(i) <= xmax)
            d = (xx(i) - xmin )/dist;
            
            if( strcmp(exactSolution,'linear'))
              exactSol(i) = exactSol(i) + slope .* d;
            end

            if(strcmp(exactSolution,'sine'))
              exactSol(i) = exactSol(i) + sin( 2.0 * freq * pi .* d);
            end
            
            if(strcmp(exactSolution,'cubic'))
              if(d <= 0.5)
                exactSol(i) =  ( (-4/3)*d + 1 )* d^2;
              else
                exactSol(i) = ( (-(4/3)*(1.0 - d)) + 1) *(1.0 - d)^2;
              end
            end

            if(strcmp(exactSolution,'quad'))
              if(d <= 0.5)
                exactSol(i) = (d-1) *d;
              else
                exactSol(i) = ( (1.0 - d)-1)* (1.0 - d);
              end
            end

            if(strcmp(exactSolution,'exp'))
              exactSol(i) = coeff * exp(-1.0/( d * ( 1.0 - d ) + 1e-100) );
            end
          end
        end

        difference = scalar - exactSol;
        L2norm     = sqrt( sum(difference.^2)/length(difference) )
        LInfinity  = max(difference)
        


        fprintf(fid,'%d %g\n',length(xx), L2norm);
%        plot(difference)
%	figure(1)
%	plot(exactSol,'k-')
%	hold on
%	plot(scalar,'b*')
%	pause;
        
      end  % timestep loop

      fclose(fid);
