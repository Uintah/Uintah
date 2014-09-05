% Give a command to server
%
function err=tstserv(cmd)

if(strcmp(cmd,'break')) return; end;% not possible to break;

transport(5,4,'127.0.0.1:5517'); % open client 
snd(cmd);                        % send command over for eval
if(strcmp(cmd,'stop')) 
 disp('shutting down the server');
 err=-1;
else
 err=rcv;                         % receive error code
 snd('break');                    % close server
end

transport(5,5);                  % close client

