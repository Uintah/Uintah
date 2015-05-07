function out(vlevel,format,varargin)
% OUT    Output to screen and file, depending on verbose level.
%
%    OUT accepts the same format of FPRINTF. Depending on the value of the global
%    variable type_outsteam, the output is directed into different streams:
%    type_outsteam = 'file'   prints the output to the (global variable)
%                    output file stream fout.
%    type_outsteam = 'screen' prints to the screen.
%    type_outsteam = 'screen' prints to both the screen and fout.
%
%    See also GLOBALPARAMS.

% Revision history:
% 16-JUN-2003    Oren Livne    Created
% 13-JUL-2005    Oren Livne    Adapted from BAM code

% Verbose levels organized as follows:
% 0     Nothing printed
% 1     High level printouts (nCells = .., error = ... for each experiment)
% 2     Function level printouts (--- adaptiveRefinement BEGIN/END ---)
% 3     Detailed printouts of internal variables in functions

globalParams;

if (vlevel > param.verboseLevel)
    return;
end

switch (param.outputType)
    case 'file',
        fprintf(param.logFile,format,varargin{:});
    case 'screen',
        fprintf(format,varargin{:});
    case 'both'
        fprintf(param.logFile,format,varargin{:});
        fprintf(format,varargin{:});
end
