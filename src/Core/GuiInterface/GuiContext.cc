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


/*
 *  GuiContext.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/GuiInterface/GuiContext.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/Util/Environment.h>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace std;
using namespace SCIRun;

GuiContext::GuiContext(GuiInterface* gui, const std::string& name, bool save)
  : gui(gui), 
    name(name),
    children(),
    cached(false),
    save(save), usedatadir(false)
{
  if (save || usedatadir) {
    const string save_flag = (save)?"1 ":"0 ";
    const string sub_flag = (usedatadir)?"1":"0";
    gui->execute("initVarStates \""+name+"\" "+save_flag+sub_flag);
  }
}

GuiContext* GuiContext::subVar(const std::string& subname, bool saveChild)
{
  save=false; // Do not save intermediate nodes
  GuiContext* child = new GuiContext(gui, name+"-"+subname, saveChild);
  children.push_back(child);
  return child;
}

void GuiContext::erase(const std::string& subname)
{
  const std::string fullname(name+"-"+subname);
  for(vector<GuiContext*>::iterator iter = children.begin();
      iter != children.end(); ++iter) 
  {
    if((*iter)->getfullname() == fullname) 
    {
      children.erase( iter );
      return;
    } 
  }
}

void GuiContext::lock()
{
  gui->lock();
}

void GuiContext::unlock()
{
  gui->unlock();
}

bool GuiContext::get(std::string& value)
{
  if(cached)
    return true;
  if(!gui->get(name, value))
    return false;
  cached=true;
  return true;
}

void GuiContext::set(const std::string& value)
{
  string tmp;
  if (save && gui->get(name, tmp) && tmp == value) { return; }
  cached=false;
  gui->set(name, value);
}

bool GuiContext::get(double& value)
{
  if(cached)
    return true;
  string result;
  if(!gui->get(name, result))
    return false;
  istringstream s(result);
  s >> value;
  if(!s)
    return false;
  cached=true;
  return true;
}

void GuiContext::set(double value)
{
  ostringstream stream;
  // Print the number 17 digits wide with decimal
  stream << showpoint << setprecision(17) << value;
  // Evaluate it in TCL to pare down extra 0's at the end
  const string svalue = gui->eval("expr "+stream.str());
  string tmp;
  if (save && gui->get(name, tmp) && tmp == svalue) { return; }
  cached=false;
  gui->set(name, svalue);
}

bool GuiContext::get(int& value)
{
  if(cached)
    return true;
  string result;
  if(!gui->get(name, result))
    return false;
  istringstream s(result);
  s >> value;
  if(!s)
    return false;
  cached=true;
  return true;
}

void GuiContext::set(int value)
{
  ostringstream val;
  val << value;
  string tmp;
  if (save && gui->get(name, tmp) && tmp == val.str()) { return; }
  cached=false;
  gui->set(name, val.str());
}

// if GuiVar has a varname like:
//    ::PSECommon_Visualization_GenStandardColorMaps_0-width
// then we want to return:
//    width
// i.e. take off everything upto and including the last occurence of _#-
//    
string GuiContext::format_varname()
{
  int state=0;
  int end_of_modulename = -1;
  //int space = 0;
  for (unsigned int i=0; i<name.size(); i++) {
    if (name[i] == ' ') return "unused";
    if (state == 0 && name[i] == '_') state=1;
    else if (state == 1 && isdigit(name[i])) state = 2;
    else if (state == 2 && isdigit(name[i])) state = 2;
    else if (state == 2 && name[i] == '-') {
      end_of_modulename = i;
      state = 0;
    } else state = 0;
  }
  if (end_of_modulename == -1)
    cerr << "Error -- couldn't format name "<< name << endl;
  return name.substr(end_of_modulename+1);
}

void GuiContext::emit(std::ostream& out,
		      const string& midx,
		      const string& indent)
{
  const string prefix = indent + (indent.length()==0?"set ":"setGlobal ");
  if(save){
    string result;
    gui->get(name, result);
    if (usedatadir && sci_getenv("SCIRUN_NET_SUBSTITUTE_DATADIR"))
    {
      const char *tmp;
      // Replace DATADIR
      if ((tmp = sci_getenv("SCIRUN_DATA")))
      {
	const string datadir(tmp);
	const string::size_type loc = result.find(datadir);
	if (loc != string::npos)
	{
	  result.replace(loc, datadir.size(), "$DATADIR");
	}
      }
      
      // Replace DATASET
      if ((tmp = sci_getenv("SCIRUN_DATASET")))
      {
	const string dataset(tmp);
	while (1)
	{
	  const string::size_type loc = result.find(dataset);
	  if (loc != string::npos)
	  {
	    result.replace(loc, dataset.size(), "$DATASET");
	  }
	  else
	  {
	    break;
	  }
	}
      }

      out << prefix << midx << "-" << format_varname() << " \""
	  << result << "\"" << std::endl;
    }
    else
    {
      if (gui->eval("isaDefaultValue {"+name+"}") == "0") {
	out << prefix << midx << "-" << format_varname() << " {"
	    << result << "}" << std::endl;
      }
    }
  }
  for(vector<GuiContext*>::iterator iter = children.begin();
      iter != children.end(); ++iter)
    (*iter)->emit(out, midx, indent);
}

void GuiContext::reset()
{
  cached=false;
  for(vector<GuiContext*>::iterator iter = children.begin();
      iter != children.end(); ++iter)
    (*iter)->reset();
}

string GuiContext::getfullname()
{
  return name;
}

GuiInterface* GuiContext::getInterface()
{
  return gui;
}

void GuiContext::dontSave()
{
  save=false;
  const string save_flag = "0 ";
  const string sub_flag = (usedatadir)?"1":"0";
  gui->execute("setVarStates \""+name+"\" "+save_flag+sub_flag);
}


void
GuiContext::setUseDatadir(bool flag)
{
  usedatadir = flag;
  const string save_flag = (save)?"1 ":"0 ";
  const string sub_flag = (usedatadir)?"1":"0";
  gui->execute("setVarStates \""+name+"\" "+save_flag+sub_flag);
}
