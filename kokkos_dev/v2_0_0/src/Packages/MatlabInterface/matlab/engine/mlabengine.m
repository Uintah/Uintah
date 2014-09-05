%
%  The contents of this file are subject to the University of Utah Public
%  License (the "License"); you may not use this file except in compliance
%  with the License.
%
%  Software distributed under the License is distributed on an "AS IS"
%  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
%  License for the specific language governing rights and limitations under
%  the License.
%
%  The Original Source Code is SCIRun, released March 12, 2001.
%
%  The Original Source Code was developed by the University of Utah.
%  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
%  University of Utah. All Rights Reserved.
%

function mlabengine(wordy,hp,cmd)

% check "transport" for existance, if not, compile

if(length(which('transport'))==0) 
 if(wordy>0) disp('mlabengine: Compiling transport.c'); end
 curdir=pwd;
 [t,r]=strtok(fliplr(which('mlabengine')),'/') ;
 engdir=fliplr(r);
 cd(engdir);
 mex transport.c ../../../src/Packages/MatlabInterface/Core/Util/bring.c
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

