
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
close all
clear all

hp=':5517';
while(1) % open server

 transport(5,3,hp);

 while(1)
  cmd=rcv;
  disp(cmd);
  if(strcmp(cmd,'break')) break; end;
  if(strcmp(cmd,'stop'))  break; end;
  err=0;
  evalin('base',cmd,'err=1;');
  snd(err);
 end

 transport(5,5,hp); % close server
 if(strcmp(cmd,'stop'))  break; end;

end

