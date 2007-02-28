%GLOBALPARAMS Global parameters for all routines in the ICE Algorithm (1D ShockTube) code.
%   Whenever a new file is added to this code, add in its first line
%   the command GLOBALPARAMS; refrain from using local variables that have a
%   name that appears in the list of GLOBALPARAMS.
%
%   See also ?.

global ...
    ...
    ghost_Left ...      % Index of left ghost cell
ghost_Right ...         % Index of right ghost cell
firstCell ...           % Index of first interior cell
lastCell ...            % Index of last interior cell
d_SMALL_NUM ...         % A small number (for bullet-proofing)
P ...                   % Parameter structure

