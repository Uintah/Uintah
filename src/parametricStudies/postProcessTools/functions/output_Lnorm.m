%______________________________________________________________________
%   This octave function opens the file and writes the header for
%   L-norm errors
%______________________________________________________________________
function output_Lnorm( fileName, variables, resolution, Lnorm )

  fid = fopen( fileName, 'a');
  [info,err,msg] = stat(fid);

  %  Write the file header
  if ( info.size == 0 )
    fprintf(fid, '# This file contains the Lnorms for the variables listed\n')
    fprintf(fid, '# X')

    for v=1:length(variables)
      fprintf( fid,' %s', variables{v} )
    end
    fprintf(fid, '\n' );
  endif

  % Write out the values
  fprintf( fid, '%i ', resolution)

  for v=1:length(variables)
    fprintf(fid,'%E ', Lnorm(v))
  end

  fprintf(fid,'\n');
  fclose(fid);
endfunction
