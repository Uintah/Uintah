//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : DynamicLoader.cc
//    Author : Martin Cole
//    Date   : Mon May 21 10:57:38 2001

#include <Core/Datatypes/DynamicLoader.h>
#include <sci_defs.h>
#include <string>
#include <Core/Util/soloader.h>
#include <Core/Util/sci_system.h>
#include <fstream>
#include <iostream>

namespace SCIRun {

using namespace std;

#ifndef ON_THE_FLY_SRC
#error ON_THE_FLY_SRC is not defined!!
#endif

#ifndef ON_THE_FLY_OBJ
#error ON_THE_FLY_OBJ is not defined!!
#endif

const string OTF_SRC_DIR(ON_THE_FLY_SRC);
const string OTF_OBJ_DIR(ON_THE_FLY_OBJ);


CompileInfo::CompileInfo(const string &fn, const string &bcn, 
			 const string &tcn, const string &tcdec) :
  filename_(fn),
  base_class_name_(bcn),

  template_class_name_(tcn),
  template_arg_(tcdec)
{
}


DynamicLoader::DynamicLoader() 
{
  algo_map_.clear();
}

DynamicLoader::~DynamicLoader() 
{
  algo_map_.clear();
}

bool
DynamicLoader::compile_and_store(const CompileInfo &info)
{
  // Try to load a .so that is already compiled
  string full_so = OTF_OBJ_DIR + string("/") + 
    info.filename_ + string("so");

  LIBRARY_HANDLE so = GetLibraryHandle(full_so.c_str());
  if (so == 0) {
    // the so does not exist.
    
    create_cc(info);
    compile_so(info.filename_);
    so = GetLibraryHandle(full_so.c_str());

    if (so == 0) { // does not compile
      cerr << "DYNAMIC COMPILATION ERROR: " << full_so 
	   << " does not compile!!" << endl;


      cerr << SOError() << endl;

      return false;
    }
  }

  typedef DynamicAlgoBase* (*maker_fun)();
  maker_fun maker = 0;
  maker = (maker_fun)GetHandleSymbolAddress(so, "maker");
  
  if (maker == 0) {
    cerr << "DYNAMIC LIB ERROR: " << full_so 
	 << " no maker function!!" << endl;
    return false;
  }
  // store this so that we can get at it again.
  store(info.filename_, DynamicAlgoHandle(maker())); 
  return true;
}

//! DynamicLoader::compile_so
//! 
//! Attempt to compile file into a .so, return true if it succeeded
//! false otherwise.
bool 
DynamicLoader::compile_so(const string& file)
{
  string command = "cd " + OTF_OBJ_DIR + "; gmake " + file + "so";

  cout << "Executing: " << command << endl;
  bool compiled =  sci_system(command.c_str()) == 0; 
  if(!compiled) {
    cerr << "DynamicLoader::compile_so() error: "
	 << "system call failed:" << endl << command << endl;
  }

  return compiled;
}



//! DynamicLoader::create_cc
//!
//! Write a .cc file, from the compile info.
bool 
DynamicLoader::create_cc(const CompileInfo &info)
{
  // Try to open the file for writing.
  string full = OTF_OBJ_DIR + "/" + info.filename_ + "cc";
  ofstream fstr(full.c_str());
  
  if (!fstr) {
    cerr << "DynamicLoader::create_cc could not create file " << full << endl;
    return false;
  }
  fstr << "// This is an autamatically generated file, do not edit!" << endl;

  // generate includes
  vector<string>::const_iterator iter = info.includes_.begin();
  while (iter != info.includes_.end()) {  
    if (*iter != "builtin")
      fstr << "#include \"" << *iter << "\"" << endl;
    ++iter;
  }

  fstr << endl;
  fstr << "using namespace SCIRun;" << endl << endl;
  fstr << "extern \"C\" {"  << endl
       << info.base_class_name_ << "* maker() {" << endl
       << "  return scinew "<< info.template_class_name_ << "<" 
       << info.template_arg_ << ">;" << endl
       << "}" << endl << "}" << endl;

  return true;
}

void 
DynamicLoader::store(const string &name, DynamicAlgoHandle algo)
{
  algo_map_[name] = algo;
}

bool 
DynamicLoader::get(const string &name, DynamicAlgoHandle algo)
{
  map_type::iterator loc = algo_map_.find(name);
  if (loc != algo_map_.end()) {
    algo = loc->second;
    return true;
  }
  // do not have this algo.
  return false;
}


} // End namespace SCIRun


