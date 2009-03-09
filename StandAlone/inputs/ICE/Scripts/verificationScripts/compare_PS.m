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
    printf('  -type  <type>       - type of passive scalar [linear|quad|cubic|exp|sine|triangular]\n')
    printf('  -vel   <num>        - Velocity of advection in m/s\n')
    printf('  -min   <num>        - minimum X value of the exact solution\n')
    printf('  -max   <num>        - maximum X value of the exact solution\n')
    printf('  -cells <\'string\'>   - This cells option is an input to the lineextract eg: \'-istart 0 0 0 -iend 99 0 0\'\n')
    printf('  -freq  <num>        - Frequency used by the sine profile\n')
    printf('  -coeff <num>        - coeff used by the quad or exp profile\n')
    printf('  -slope <num>        - slope of the linear/triangular profile\n')
    printf('  -L                  - Compute L2 error for last timestep only (useful for testing framework)\n')
    printf('  -o <fname>          - Dump the output (L2Error) to a file\n')
    printf('  -plot <true, false> - produce a plot \n') 
    printf('  -pDir <1,2,3>       - principal direction \n')
    end 
%      argv = {"-uda";"advectCubic__r100v10.uda.001";"-type"; "cubic";"-vel";"10";"-min";"-0.1";"-max";"0.1";"-cells";"-istart 0 0 0 -iend 99 0 0";"-coeff";"10";"-L";"-o";"test_out.txt"}

 %     nargin = length(argv)
      
      %__________________________________
      % default user inputs
      makePlot = false;
      pDir = 1;
      startEnd = "";

      if (nargin == 0)
         Usage
	 exit
      endif 

      last = false;

      for i = 1:nargin
	
	%________________________________
	% USER INPUTS
	
	
	if strcmp(sprintf("%s",argv(i,:)),"-uda")
	  uda = sprintf("%s",argv(++i,:))
	elseif strcmp(sprintf("%s",argv(i,:)),"-type")
	  exactSolution = sprintf("%s",argv(++i,:))
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
	elseif strcmp(sprintf("%s",argv(i,:)),"-plot")
	  makePlot = sprintf("%s",argv(++i,:))
       elseif strcmp( sprintf("%s",argv(i,:)),"-pDir")
         pDir = str2num("%s",argv(++i,:)) 
	endif

      endfor
      
      %________________________________
      % open data files
      datFile = strrep(uda,"uda","dat");
      fid     = fopen(output_file, 'w');
      fid2    = fopen(datFile,'w');
      

      %________________________________
      %  extract the physical time
      c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
      [status0, result0]=unix(c0);
      physicalTime  = load('tmp');
      nDumps = length(physicalTime);
      startVal = 1;

      if (last)
	startVal = nDumps;
      end


      %_______________________________
      % is the domain periodic
      c1 = sprintf('puda -gridstats %s | grep -m 1 Periodic | tr -d "[:alpha:][:punct:]" >& tmp',uda);
      [status1, result1]=unix(c1);
      periodic = load('tmp');
      %_________________________________
      % Loop over all the timesteps
      for(n = startVal:nDumps )
	%  n = input('input timestep') 
	ts = n-1;
	
	time = sprintf('%16.15E sec',physicalTime(n))
	
       if( strcmp(startEnd,"") )
         if(pDir == 1)
           S_E = sprintf('-istart 0 0 0 -iend 1000000000 0 0')
         end
         if(pDir == 2)
           S_E = sprintf('-istart 0 0 0 -iend 0 1000000000 0')
         end
         if(pDir == 3)
           S_E = sprintf('-istart 0 0 0 -iend 0 0 1000000000')
         end
         oneZero = 1 - periodic(pDir);
       else
         S_E = startEnd;
         oneZero = 0;
       end
	
	unix('/bin/rm -f scalar-f.dat');

	%____________________________
	%   scalar-F
        c6 = sprintf('lineextract -v scalar-f -l 0 -cellCoords -timestep %i %s -o scalar-f.dat -m 0  -uda %s',ts,S_E,uda)

        [s6, r6]=unix(c6);
        scalarArray = load('scalar-f.dat');
        
        % ignore the ghost cells
        len    = length(scalarArray(:,1));
        xx     = scalarArray(1:len-oneZero,1);
        scalar = scalarArray(1:len-oneZero,4);
        %_________________________________
        % Exact Solution on each level
        dist   = exactSolMax - exactSolMin;
        offset = (physicalTime(n)) * velocity
        xmin   = exactSolMin + offset
        xmax   = exactSolMax + offset
                
        fuzz_min = abs(xmin * 2.22e-16);
        fuzz_max = abs(xmax * 2.22e-16);
        xmin_fuzz = xmin - fuzz_min;
        xmax_fuzz = xmax + fuzz_max;
        
        d = xx * 0;
        exactSol=xx * 0;

        for( i = 1:length(xx))
          
          if( xx(i) >= (xmin-fuzz_min) && xx(i)  <= (xmax+fuzz_max))
          
            d(i) = (xx(i) - xmin )/dist;
            
            if (d(i) < -fuzz_min || d(i) > (1.0+fuzz_max) )
              disp('Warning something has gone wrong')
              i
              xx(i)
              xmin
              dist
              xx(i) - xmin
            end
            
            if( strcmp(exactSolution,'linear'))
              exactSol(i) = exactSol(i) + slope .* d(i);
            end

            if( strcmp(exactSolution,'triangular'))
	      if (d(i) <= 0.5)
                exactSol(i) = exactSol(i) + slope .* d(i);
              else
                exactSol(i) = exactSol(i) + slope .* (1.0-d(i));
              end
            end


            if(strcmp(exactSolution,'sine'))
              exactSol(i) = exactSol(i) + sin( 2.0 * freq * pi .* d(i));
            end
            
            if(strcmp(exactSolution,'cubic'))
              if(d <= 0.5)
                exactSol(i) =  ( (-4/3)*d(i) + 1 )* d(i)^2;
              else
                exactSol(i) = ( (-(4/3)*(1.0 - d(i))) + 1) *(1.0 - d(i))^2;
              end
            end

            if(strcmp(exactSolution,'quad'))
              if(d <= 0.5)
                exactSol(i) = (d(i)-1) *d(i);
              else
                exactSol(i) = ( (1.0 - d(i))-1)* (1.0 - d(i));
              end
            end

            if(strcmp(exactSolution,'exp'))
              exactSol(i) = coeff * exp(-1.0/( d(i) * ( 1.0 - d(i) ) + 1e-100) );
            end
          end
        end
        
        for( i = 1:length(xx))
          difference(i) = scalar(i) - exactSol(i);
          fprintf(fid2,'%16.15E %16.15E %16.15E %16.15E\n',xx(i),difference(i),scalar(i), exactSol(i));
        end
        
        lenExactXol= length(exactSol)
        N          = length(difference)
        L2norm     = sqrt( sum(difference.^2)/N)
        LInfinity  = max(difference)

        fprintf(fid,'%16.16g\n',L2norm);
        
        if (strcmp(makePlot,"true"))
          
          plot(xx,difference,'bo;difference;')
          xlabel('Position')
          grid on
          
          figure(1)
          plot(d,exactSol,'k-;exactSol;',d,scalar,'b*;Simulation Results;')
          xlabel('non-dimensional Position')
          grid on
          
          figure(2)
          plot(xx,exactSol,'k-;exactSol;', xx,scalar,'b*;Simulation Results;')
          xlabel('Position')
          grid on
          
          pause
       endif
        
      end  % timestep loop

      % clean up 
      fclose(fid);
      fclose(fid2);
