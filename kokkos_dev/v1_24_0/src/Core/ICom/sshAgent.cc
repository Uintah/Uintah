/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  sshAgent.cc: 
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */


#include <Core/ICom/sshAgent.h>
#include <Core/Containers/StringUtil.h>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
 
using namespace ICom;
using namespace std;
using namespace SCIRun;

int sshagentpid = 0;

bool sshAgent::startagent() {

  // Sorry, no windows equivalent for this ones
#ifndef _WIN32
  bool foundkeyfile;
		
  // If no key_file was specified, we assume the user is not interested
  // in ssh as a tool to get through firewalls
  if (!(sci_getenv("SSH_KEY_FILE"))) {
    if ((sci_getenv("SSH_AGENT_PID")) && (sci_getenv("SSH_AUTH_SOCK"))) {
      // If someone already did all the work of setting up the agent
      // for making remote connections and no keyfile has been
      // specified, we will try to use the default settings. It is not
      // clear whether a connection can be made this way, but it will
      // be tried
      cout << "Starting ssh-agent... detected the presence of an ssh_agent: ssh connections will be enabled\n";
      return true;
    }
    return false;
  }
	
  cout << "Starting ssh-agent...";
	
  if (!(sci_getenv("SCIRUN_OBJDIR"))) {
    cerr << "The SCIRUN_OBJDIR environment has not been set\n";
  }
	
  string keyfile = sci_getenv("SSH_KEY_FILE");
	
  // 1. Assume the user is did include the full path
  foundkeyfile = iskeyfile(keyfile);
	
  // 2. Test HOME/.ssh
  if (!foundkeyfile) {
    if (getenv("HOME")) {
      string home = getenv("HOME");
      keyfile = home + "/.ssh/" + keyfile;
      foundkeyfile = iskeyfile(keyfile);
    }
  }
	
  // 3. Test HOME
  if (!foundkeyfile) {
    if (getenv("HOME")) {
      string home = getenv("HOME");
      keyfile = home + "/" + keyfile;
      foundkeyfile = iskeyfile(keyfile);
    }
  }
  
  // 4. Test OBJDIR
  if (!foundkeyfile) {
    if (sci_getenv("SCIRUN_OBJDIR")) {
      string objdir = getenv("OBJDIR");
      keyfile = objdir + "/" + keyfile;
      foundkeyfile = iskeyfile(keyfile);
    }
  }	
	
  // 5. Any other directory is not checked, the user should put it in
  // HOME/.ssh where keyfiles are supposed to be stored
	
  string scirun_objdir = sci_getenv("SCIRUN_OBJDIR");
  string agentfile = scirun_objdir + "/remote/ssh/.agent";
	
  string cmd = "ssh-agent -s >" + agentfile + "; source " + agentfile +
    "; ssh-add " + keyfile;
  system(cmd.c_str()); 

  FILE *agentfid;
	
  if (!(agentfid = fopen(agentfile.c_str(),"r"))) {
    cerr << "failed\n";
    return false;
  }
	
  char buffer[256];	
  while(fgets(buffer,255,agentfid)) {
    if (strncmp("SSH_AUTH_SOCK=",buffer,14)==0) {
      size_t size = strlen(buffer);
      for (size_t p = 14;p<size;p++) {
        if(buffer[p] == ';') {
          buffer[p] = '\0'; break;
        }
      }
      string sciauthsock = &(buffer[14]);
      sci_putenv("SSH_AUTH_SOCK",sciauthsock);
    }
    if (strncmp("SSH_AGENT_PID=",buffer,14)==0) {
      size_t size = strlen(buffer);
      for (size_t p = 14;p<size;p++) {
        if(buffer[p] == ';') {
          buffer[p] = '\0';
          break;
        }
      }
      string sciagentpid = &(buffer[14]);
      sci_putenv("SSH_AGENT_PID",sciagentpid);
    }		
  }
  fclose(agentfid);
  

  if ((sci_getenv("SSH_AGENT_PID")) && (sci_getenv("SSH_AUTH_SOCK"))) {
    // if we created the agent we should clear it before exiting
    // SCIRun
    string sshpid = sci_getenv("SSH_AGENT_PID");
    string_to_int(sshpid.c_str(),sshagentpid);
    atexit(sshAgent::killagent);
    cout << "ssh-agent started\n";
    return true;
  }
	
  cout << "failed\n";
#endif
  return false;
}

bool sshAgent::iskeyfile(string str)
{
  FILE * fid = NULL;

  if ( (fid = fopen(str.c_str(),"r")) != NULL ) {
    fclose(fid);
    return true;
  } else {
    return false;
  }
}


void sshAgent::killagent()
{
  // Sorry, no windows equivalent for this one
#ifndef _WIN32
  if(sshagentpid)
    kill(sshagentpid,SIGTERM);
#endif
}
