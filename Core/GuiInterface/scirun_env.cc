/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <sci_defs.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/PackageDB.h>
#include <Dataflow/Network/Scheduler.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <Core/Thread/Thread.h>
#include <stdio.h>
#include <Core/Util/RWS.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Core/GuiInterface/scirun_env.h>
#include <Core/GuiInterface/TCLInterface.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>



namespace SCIRun {

char* MacroSubstitute(char* var_val)
{
  int cur = 0;
  int start = 0;
  int end = start;
  string newstring("");
  char* macro = 0;
  
  if (var_val==0)
    return 0;

  int length = (int)strlen(var_val);

  while (cur < length-1) {
    if (var_val[cur] == '$' && var_val[cur+1]=='(') {
      cur+=2;
      start = cur;
      while (cur < length) {
	if (var_val[cur]==')') {
	  end = cur-1;
	  var_val[cur]='\0';
	  macro = new char[end-start+1];
	  sprintf(macro,"%s",&var_val[start]);
	  char *env = getenv(macro);
	  delete[] macro;
	  if (env) 
	    newstring += string(env);
	  var_val[cur]=')';
	  cur++;
	  break;
	} else
	  cur++;
      }
    } else {
      newstring += var_val[cur];
      cur++;
    }
  }

  newstring += var_val[cur]; // don't forget the last character!

  unsigned long newlength = strlen(newstring.c_str());
  char* retval = new char[newlength+1];
  sprintf(retval,"%s",newstring.c_str());
  
  return retval;
}


void sci_putenv(const string &var,const string &val,
		GuiInterface *gui, bool force)
{
  // Check TCL's backup of the enviroment when scirun started
  // to make sure we dont overwrite any env string that the user
  // set before running
  if (!force && gui) {
    string result;
    gui->eval("info exists alreadySetEnv", result);
    if (result == "1") {
      gui->eval("lsearch $alreadySetEnv "+var,result);      
      if (result != "-1") return;
    }
  }

  const string envarstr = var+"="+val;
  char *envar = scinew char[envarstr.size()+1];
  memcpy(envar, envarstr.c_str(), envarstr.size());
  envar[envarstr.size()] = '\0';
  putenv(envar);
  // Set the var in tcl too
  if (gui) gui->execute("global env; set env("+var+") {"+val+"}");
}  

bool RCParse(const char* rcfile, GuiInterface *gui)
{
  FILE* filein = fopen(rcfile,"r");
  char var[0xff];
  char var_val[0xffff];

  if (!filein)
    return false;

  int i=1; // prevent warning

  while (i) {
    var[0]='\0';
    var_val[0]='\0';
    if (fscanf(filein,"%[^=]=%s",var,var_val)!=EOF){
      if (var[0]!='\0' && var_val[0]!='\0') {
	removeLTWhiteSpace(var);
	removeLTWhiteSpace(var_val);
	char* sub = MacroSubstitute(var_val);
	sci_putenv(var,sub,gui);
	delete[] sub;
      }
    } else
      break;
  }

  fclose(filein);

  return true;
}


void parse_scirunrc(GuiInterface *gui)
{
  gui->execute("set alreadySetEnv [array names env]");
  ostringstream str;
  
  bool foundrc=false;
  str << "Parsing .scirunrc... ";
  
  // check the local directory
  foundrc = RCParse(".scirunrc",gui);
  if (foundrc)
    str << "./.scirunrc" << std::endl;
  
  // check the BUILD_DIR
  if (!foundrc) {
    foundrc = RCParse((SCIRUN_OBJDIR+string("/.scirunrc")).c_str(),gui);
    if (foundrc)
      str << SCIRUN_OBJDIR << "/.scirunrc" << std::endl;
  }
  
  // check the user's home directory
  if (!foundrc) {
    char* HOME = getenv("HOME");
    
    if (HOME) {
      string home(HOME);
      home += "/.scirunrc";
      foundrc = RCParse(home.c_str(),gui);
      if (foundrc)
	str << home << std::endl;
    }
  }
  
  // check the INSTALL_DIR
  if (!foundrc) {
    foundrc = RCParse((SCIRUN_SRCDIR+string("/.scirunrc")).c_str(),gui);
    if (foundrc)
      str << SCIRUN_SRCDIR << "/.scirunrc" << std::endl;
  }
  
  // Since the dot file is optional report only if it was found.
  if( foundrc ) {
    cout << str.str();
  }
  else {
    // check to make sure home directory is writeable.
    char* HOME = getenv("HOME");
    if (HOME) {
      string homerc = string(HOME) + "/.scirunrc";
      int fd;
      if ((fd = creat(homerc.c_str(), S_IREAD | S_IWRITE)) != -1)
      {
	close(fd);
	unlink(homerc.c_str());
	string tclresult;
	gui->eval("licenseDialog 1", tclresult);
	if (tclresult == "cancel")
	{
	  Thread::exitAll(1);
	}
	else if (tclresult == "accept")
	{
	  if ((fd = creat(homerc.c_str(), S_IREAD | S_IWRITE)) != -1)
	  {
	    close(fd);
	  }
	}
      }
    }	  
  }
}



} // namespace SCIRun 
