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

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/Util/Environment.h>
#include <Core/Util/Assert.h>
#include <Core/Math/MiscMath.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <sstream>
#include <iomanip>

using std::string;
using std::vector;
using namespace SCIRun;

GuiContext::GuiContext(GuiInterface* gui, const std::string& name, bool save, GuiContext *parent)
  : gui(gui),
    parent(parent),
    name(name),
    children(),
    context_state(CACHE_E)
{
  context_state |= save?SAVE_E:0;
  if (save)
    tcl_setVarStates();
}

//GuiContext::GuiContext()
//{
//}


GuiContext::~GuiContext()
{
  dontSave();
  if (parent) {
    for(vector<GuiContext*>::iterator iter = parent->children.begin();
	iter != parent->children.end(); ++iter) 
    {
      if (*iter == this)
      {
	parent->children.erase(iter);
	return;
      } 
    }
  }
}



GuiContext* GuiContext::subVar(const std::string& subname, bool saveChild)
{
  dontSave(); // Do not save intermediate nodes
  GuiContext* child = scinew GuiContext(gui, name+"-"+subname, saveChild, this);
  children.push_back(child);
  return child;
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
  if((context_state & CACHE_E) && (context_state & CACHED_E))
    return true;
  context_state &= ~CACHED_E;
  if(!getString(name, value))
    return false;
  context_state |= CACHED_E;
  return true;
}

void GuiContext::set(const std::string& value)
{
  string tmp;
  if ((context_state & SAVE_E) && getString(name, tmp) && tmp == value) { 
    return; 
  }
  context_state &= ~CACHED_E;
  setString(name, value);
}

bool GuiContext::get(double& value)
{
  if((context_state & CACHE_E) && (context_state & CACHED_E))
    return true;
  context_state &= ~CACHED_E;
  string result;
  if(!getString(name, result))
    return false;
  istringstream s(result);
  s >> value;
  if(!s)
    return false;
  context_state |= CACHED_E;
  return true;
}

void GuiContext::set(double value)
{
  value = MakeReal(value);
  ostringstream stream;
  // Print the number 17 digits wide with decimal
  stream << setiosflags(ios::showpoint) << setprecision(17) << value;
  // Evaluate it in TCL to pare down extra 0's at the end
  const string svalue = gui->eval("expr "+stream.str());
  string tmp;
  if ((context_state & SAVE_E) && getString(name, tmp) && tmp == svalue) { 
    return; 
  }
  context_state &= ~CACHED_E;
  gui->set(name, svalue);
}

bool GuiContext::get(int& value)
{
  if ((context_state & CACHE_E) && (context_state & CACHED_E))
    return true;
  context_state &= ~CACHED_E;
  string result;
  if(!getString(name, result))
    return false;
  istringstream s(result);
  s >> value;
  if(!s)
    return false;
  context_state |= CACHED_E;
  return true;
}

void GuiContext::set(int value)
{
  ostringstream val;
  val << value;
  string tmp;
  if ((context_state & SAVE_E) && gui->get(name, tmp) && tmp == val.str()) { 
    return; 
  }
  context_state &= ~CACHED_E;
  setString(name, val.str());
}

// if GuiVar has a varname like:
//    ::PSECommon_Visualization_GenStandardColorMaps_0-width
// then we want to return:
//    width
// i.e. take off everything upto and including the last occurence of _#-
//    
string GuiContext::getName()
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


// if GuiVar has a varname like:
//    ::PSECommon_Visualization_GenStandardColorMaps_0-width
// then we want to return:
// ::PSECommon_Visualization_GenStandardColorMaps_0
// i.e. everything upto but not including the last occurence of "-"
//    
string GuiContext::getPrefix()
{
  int state=0;
  int end_of_modulename = 0;
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
  if (!end_of_modulename)
    cerr << "Error -- couldn't format name "<< name << endl;
  return name.substr(0,end_of_modulename);
}


string GuiContext::getMapKeyFromString(const string &varname) {
  ASSERT(stringIsaMap(varname));
  ASSERT(varname.size() >= 4); // smallest possible name: ie a(b) = 4 chars
  string::size_type open_paren = varname.find_first_of("(");
  string::size_type close_paren = varname.find_last_of(")");
  ASSERT(open_paren && close_paren);
  string key = varname.substr(open_paren+1, close_paren-open_paren-1);
  if (stringIsaMap(key) || stringIsaListElement(key))
  {
    string new_key;
    if (!getString(key, new_key)) {
      //      cerr << "Cannot figure value of key: " << key 
      //	   << " of var: " << varname << std::endl;
      return key;
    }
    key = new_key;
  }
  return key;
}



string GuiContext::getMapNameFromString(const string &varname) {
  ASSERT(stringIsaMap(varname));
  ASSERT(varname.size() >= 4); // smallest possible name: ie a(b) = 4 chars
  string::size_type open_paren = varname.find_first_of("(");
  ASSERT(open_paren);
  return varname.substr(0, open_paren);
}

int GuiContext::popLastListIndexFromString(string &varname) {
  ASSERT(varname.size() >= 4); // smallest possible name: ie i[0] = 4 chars
  string::size_type open_bracket = varname.find_last_of("[");
  string::size_type close_bracket = varname.find_last_of("]");
  ASSERT(open_bracket && close_bracket);
  int i = -1;
  string_to_int(varname.substr(open_bracket+1, close_bracket-1),i);
  if (open_bracket > 0)
    varname = varname.substr(0, open_bracket);
  return i;
}



bool GuiContext::getString(const std::string& varname, std::string& value) {
  bool success = false;
  if (stringIsaListElement(varname))
  {
    vector<int> indexes;
    string listname = varname;
    int idx;
    while((idx = popLastListIndexFromString(listname)) != -1)
      indexes.push_back(idx);

    string list_contents;
    if (!getString(listname, list_contents)) {
      //      cerr << "Cannot find list variable: " << listname;
      return false;
    }
    success = gui->extract_element_from_list (list_contents, indexes, value);
    //    if (!success) cerr << "Cannont find List Element: " << varname << std::endl;
    return success;
  } else if (stringIsaMap(varname)) {
    success = gui->get_map(getMapNameFromString(varname), 
			   getMapKeyFromString(varname), 
			   value);
    //    if (!success) cerr << "Cannot find Map Element: " << varname << std::endl;
    return success;
  }
  // else? just do a standard gui get
  success = gui->get(varname, value);
  return success;
}


bool GuiContext::setString(const std::string& varname, const std::string& value) {
  bool success = true;
  if (stringIsaListElement(varname))
  {
    vector<int> indexes;
    string listname = varname;
    int idx;
    while((idx = popLastListIndexFromString(listname)) != -1)
      indexes.push_back(idx);

    string list_contents;
    if (!getString(listname,list_contents)) {
      //      cerr << "Cannot find list variable: " << listname;
      return false;
    }
    success = gui->set_element_in_list(list_contents, indexes, value);
    //    if (!success) cerr << "Cannont find List Element: " << varname << std::endl;
    success = setString(listname, list_contents);
    //    if (!success) 
    //  cerr << "Cannont set list: " << listname 
    //	   << " to " << list_contents << std::endl;

    return success;
  } else if (stringIsaMap(varname)) {
    success = gui->set_map(getMapNameFromString(varname), 
			   getMapKeyFromString(varname), 
			   value);
    //    if (!success) cerr << "Cannot set Map Element: " << varname << std::endl;
    return success;
  }
  // else just do a standard gui set
  gui->set(varname, value);
  return success;
}

bool GuiContext::stringIsaListElement(const string &str) {
  if (!str.length()) 
    return false;
  return (str[str.length()-1] == ']');
}

bool GuiContext::stringIsaMap(const string &str) {
  if (!str.length()) 
    return false;
  return (str[str.length()-1] == ')');
}




void GuiContext::reset()
{
  context_state &= ~CACHED_E;
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

void GuiContext::tcl_setVarStates() {
  const string save_flag = (context_state & SAVE_E)?"1":"0";
  const string sub_flag = (context_state & SUBSTITUTE_DATADIR_E)?" 1":" 0";
  gui->execute("setVarStates \""+name+"\" "+save_flag+sub_flag);
}

void
GuiContext::dontSave()
{
  if ((context_state & SAVE_E) == 0) return;
  context_state &= ~SAVE_E;
  tcl_setVarStates();
}

void
GuiContext::doSave()
{
  if ((context_state & SAVE_E) == SAVE_E) return;
  context_state |= SAVE_E;
  tcl_setVarStates();
}

void
GuiContext::setUseDatadir(bool flag)
{
  if (flag) {
    if ((context_state & SUBSTITUTE_DATADIR_E) == SUBSTITUTE_DATADIR_E)
      return;
    context_state |= SUBSTITUTE_DATADIR_E;
  }
  else {
    if ((context_state & SUBSTITUTE_DATADIR_E) == 0)
      return;
    context_state &= ~SUBSTITUTE_DATADIR_E;
  }
  tcl_setVarStates();
}


