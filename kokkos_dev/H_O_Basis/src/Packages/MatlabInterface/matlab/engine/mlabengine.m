%
%  For more information, please see: http://software.sci.utah.edu
% 
%  The MIT License
% 
%  Copyright (c) 2004 Scientific Computing and Imaging Institute,
%  University of Utah.
% 
%  License for the specific language governing rights and limitations under
%  Permission is hereby granted, free of charge, to any person obtaining a
%  copy of this software and associated documentation files (the "Software"),
%  to deal in the Software without restriction, including without limitation
%  the rights to use, copy, modify, merge, publish, distribute, sublicense,
%  and/or sell copies of the Software, and to permit persons to whom the
%  Software is furnished to do so, subject to the following conditions:
% 
%  The above copyright notice and this permission notice shall be included
%  in all copies or substantial portions of the Software.
% 
%  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
%  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
%  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
%  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
%  DEALINGS IN THE SOFTWARE.
%


function mlabengine(wordy,hp,cmd)

% check "transport" for existance, if not, compile

if(length(which('transport'))==0) 
 if(wordy>0) disp('mlabengine: Compiling transport.c'); end
 curdir=pwd;
 [t,r]=strtok(fliplr(which('mlabengine')),'/') ;
 engdir=fliplr(r);
 cd(engdir);
 mex transport.c bring.c
 cd(curdir);
end

% check for wordy, default 0

if(~exist('wordy'))
 wordy=0;
elseif(length(wordy)==0)
 wordy=0;
end

global wrd;
wrd=wordy;

% check for hp, default 5517

if(~exist('hp')) 
 hp='127.0.0.1:5517';
elseif(length(hp)==0) 
 hp='127.0.0.1:5517';
end

% check for command, if yes, send & return

if(exist('cmd','var')) 
 tstserv(wordy,hp,cmd); 
 return;
end

% start listening, open the server

while(1)

 transport(wordy,3,hp);

 while(1)
  cmd=rcv;
  if(wordy>0) disp([ 'mlabengine: ' cmd ]); end;
  if(strcmp(cmd,'break')) break; end;
  if(strcmp(cmd,'stop'))  break; end;
  err=0;
  evalin('base',cmd,'err=1;');
  snd(err);
 end

 transport(wordy,5,hp); % close server
 if(strcmp(cmd,'stop'))  break; end;

end

return;

% function receive

function a=rcv;
global wrd;
a=transport(wrd,1,'');

% function send

function snd(a);
global wrd;
transport(wrd,2,'',a);

% Give a command to server
%
function err=tstserv(wordy,hp,cmd)

if(strcmp(cmd,'break')) return; end;% not possible to break;

transport(wordy,4,hp);           % open client 
snd(cmd);                        % send command over for eval
if(strcmp(cmd,'stop')) 
 if(wordy>0) disp('mlabengine: tstserv: shutting down the server'); end;
 err=-1;
else
 err=rcv;                        % receive error code
 snd('break');                   % close server
end

transport(wordy,5);              % close client

return;

