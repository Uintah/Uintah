function [of] = outputFunctions()
  
  of.writeData        = @writeData;
  of.writeProbePoints = @writeProbePoints
  %______________________________________________________________________
  function writeData( filename, time, data)

    fid = fopen(filename, 'w');
    fprintf(fid,'time %15.16E\n',time)
        
    % write out the file header
    f_names = fieldnames(data);           % find all of the field names in the data struct
    
    for c=1:length(f_names)              % loop over all of the fields in the struct
      fn = f_names{c};
      fprintf(fid,'%s  ',fn);
    end
    fprintf(fid,'\n');
    
    % write out the data
    f_names(1);
    X = getfield(data,f_names{1});
    
    for c=1:length(X)
      
      for f=1:length(f_names)
        var = getfield(data,f_names{f},{c});      % get the field from the main array;
        fprintf(fid,'%16.15E  ',var);
      end
      fprintf(fid,'\n');
      
    end
    fclose(fid);

  end
  
  
  %______________________________________________________________________
  function writeProbePoints(filename, tstep, time, data, cell)
  
    if( tstep == 1) 
      fid = fopen(filename, 'w');         % overwrite file
      fprintf( 'now opening: %s \n',filename);
    else  
      fid = fopen(filename, 'a');         % append file
    end
    
    f_names = fieldnames(data);           % find all of the field names in the data struct   
       
    % write out the file header on first timestep  
    if( tstep == 1)    
      fprintf(fid,'Time  ');
      
      for c=1:length(f_names)            % loop over all of the fields in the struct
        fn = f_names{c};
        fprintf(fid,'%s  ',fn);
      end
    
      fprintf(fid,'\n');
    end
    
    % write out the data
    fprintf(fid,'%16.15E  ',time);
    
    for f=1:length(f_names)
      var = getfield(data,f_names{f},{cell});      % get the field from the main array;
      fprintf(fid,'%16.15E  ',var);
    end
    
    fprintf(fid,'\n');
    fclose(fid);

  end
end
