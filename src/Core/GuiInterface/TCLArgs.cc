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
 *  TCLArgs.cc: Interface to TCL
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiManager.h>
#include <Core/GuiInterface/TCLArgs.h>
#include <iostream>
using std::cerr;
using std::endl;
using std::ostream;
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>                             // defines read() and write()
#endif


namespace SCIRun {

 

TCLArgs::TCLArgs(int argc, char* argv[])
  : args_(argc)
{
  for(int i=0;i<argc;i++)
    args_[i] = string(argv[i]);
  have_error_ = false;
  have_result_ = false;
}

TCLArgs::~TCLArgs()
{
}

int 
TCLArgs::count()
{
  return args_.size();
}

string 
TCLArgs::operator[](int i)
{
  return args_[i];
}

void 
TCLArgs::error(const string& e)
{
  string_ = e;
  have_error_ = true;
  have_result_ = true;
}

void 
TCLArgs::result(const string& r)
{
  if(!have_error_){
    string_ = r;
    have_result_ = true;
  }
}

void 
TCLArgs::append_result(const string& r)
{
  if(!have_error_){
    string_ += r;
    have_result_ = true;
  }
}

void 
TCLArgs::append_element(const string& e)
{
  if(!have_error_){
    if(have_result_)
      string_ += ' ';
    string_ += e;
    have_result_ = true;
  }
}

string 
TCLArgs::make_list(const string& item1, const string& item2)
{
  char* argv[2];
  argv[0]= ccast_unsafe(item1);
  argv[1]= ccast_unsafe(item2);
  char* ilist=gm->merge(2, argv);
  string res(ilist);
  free(ilist);
  return res;
}

string 
TCLArgs::make_list(const string& item1, const string& item2,
		   const string& item3)
{
  char* argv[3];
  argv[0]=ccast_unsafe(item1);
  argv[1]=ccast_unsafe(item2);
  argv[2]=ccast_unsafe(item3);
  char* ilist=gm->merge(3, argv);
  string res(ilist);
  free(ilist);
  return res;
}

string 
TCLArgs::make_list(const vector<string>& items)
{
  char** argv=scinew char*[items.size()];
  for(unsigned int i=0; i<items.size(); i++)
    argv[i]= ccast_unsafe(items[i]);
  char* ilist=gm->merge(items.size(), argv);
  string res(ilist);
  free(ilist);
  delete[] argv;
  return res;
}



} // End namespace SCIRun

