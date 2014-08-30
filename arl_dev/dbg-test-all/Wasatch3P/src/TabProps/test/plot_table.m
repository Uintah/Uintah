function plot_table(mfileName)
%plot_table( mfileName )
% Example:
%  plot_table( 'MixingModel' );  % for a m-file named "MixingModel.m"

% jcs old interface:
% Example:
%  MixingModel
%  plot_table( ivar, ivarNames, dvar, dvarNames, npts );
% This would allow faster repeat queries since it would not force a re-load
% of the table each time.


eval(mfileName);

nvar = length(npts); % number of independent variables

% select the dependent variable to plot
ok=0;
while ~ok
  [myDVAR,ok] = listdlg( ...
    'ListString',dvarNames,...
    'SelectionMode','single', ...
    'PromptString','Select the dependent variable' ...
    );
end

% perform 1D or 2D plots, depending on # of independent variables.
switch nvar
  
  case 1
    plot( ivar{1}, dvar{myDVAR} );
    xlabel(ivarNames{1});
    ylabel(dvarNames{myDVAR});
  
  case 2
    x1 = reshape(ivar{1},npts(1),npts(2));
    x2 = reshape(ivar{2},npts(1),npts(2));
    y  = reshape(dvar{myDVAR},npts(1),npts(2));
    surf( x1,x2,y );
    shading interp; 
    xlabel(ivarNames{1});
    ylabel(ivarNames{2});
    zlabel(dvarNames{myDVAR});

    % show a series of lines equivalent to the surface plot
    figure; hold on 
    plot(x1,y(:,1),'g--','LineWidth',1.0);
    for i = 2:size(y,2)-1
      plot(x1,y(:,i),'k-','LineWidth',1.0);
    end
    plot(x1,y(:,size(y,2)),'r--','LineWidth',1.0);
    xlabel(ivarNames{1});
    ylabel(dvarNames{myDVAR});
    grid on; hold off

  
  case 3
    ok=0;
    while ~ok
      [vars,ok] = listdlg( ...
        'ListString',ivarNames, ...
        'SelectionMode','multiple',...
        'PromptString','Select two independent variables for visualization' ...
        );
      if ok
        ok = length(vars)==2;
      end
    end
    extraVarIx = setdiff(1:3,vars);
    msg = sprintf('enter the value for %s: ',ivarNames{extraVarIx});
    loc = input(msg);
    
    extraVar = ivar{ extraVarIx };
    
    active = [0 0 0];
    active(vars)=1;
    
    t1 = downselect_3d( extraVar, active, npts );
    t2 = abs(loc-t1);
    ix = find(t2==min(t2));
    if length(ix)>1
      ix = ix(1);
    end
    fprintf('Generating plot at %s = %g\n',ivarNames{extraVarIx},t1(ix));
    
    x1 = select_variable_3d( ivar{vars(1)}, active, npts, ix );
    x2 = select_variable_3d( ivar{vars(2)}, active, npts, ix );
    y  = select_variable_3d( dvar{myDVAR},  active, npts, ix );
    
    surf(x1,x2,y);
    shading interp;
    xlabel(ivarNames{vars(1)});
    ylabel(ivarNames{vars(2)});
    zlabel(dvarNames{myDVAR});

    % show a series of lines equivalent to the surface plot
    figure;  hold on 
    plot(x1,y(:,1),'g--','LineWidth',1.0)
    for i = 2:size(y,2)-1
      plot(x1,y(:,i),'k-','LineWidth',1.0)
    end
    plot(x1,y(:,size(y,2)),'r--','LineWidth',1.0)
    xlabel(ivarNames{vars(1)});
    ylabel(dvarNames{myDVAR});
    grid on;  hold off

  otherwise
    error('unsupported number of independent variables');
end

end


function var = select_variable_3d( var, active, dim, ix )

assert( length(ix)    ==1 );
assert( length(active)==3 );
assert( length(dim)   ==3 );

var = reshape(var,dim(1),dim(2),dim(3));

if active(1)
  if active(2)
    var = squeeze(var(:,:,ix));
  else
    var = squeeze(var(:,ix,:));
  end
else
  var = squeeze(var(ix,:,:));
end

end

function var = downselect_3d( var, active, dim )

var = reshape(var,dim(1),dim(2),dim(3));

if active(1)
  if active(2)
    var = squeeze(var(1,1,:));
  else
    var = squeeze(var(1,:,1));
  end
else
  var = squeeze(var(:,1,1));
end

end
