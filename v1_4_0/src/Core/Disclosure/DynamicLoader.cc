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

#include <Core/Disclosure/DynamicLoader.h>
#include <sci_defs.h>
#include <string>
#include <Core/Util/soloader.h>
#include <Core/Util/sci_system.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

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

DynamicLoader* DynamicLoader::scirun_loader_ = 0;
Mutex DynamicLoader::scirun_loader_lock_("DynamicLoader: static instance");

CompileInfo::CompileInfo(const string &fn, const string &bcn, 
			 const string &tcn, const string &tcdec) :
  filename_(fn),
  base_class_name_(bcn),
  template_class_name_(tcn),
  template_arg_(tcdec)
{
}


void
CompileInfo::add_include(const string &inc)
{
  //std::remove(includes_.begin(), includes_.end(), inc);
  includes_.push_front(inc);
}


DynamicAlgoBase::DynamicAlgoBase() :
  ref_cnt(0),
  lock("DynamicAlgoBase ref_cnt lock")
{
}

DynamicAlgoBase::~DynamicAlgoBase()
{
}


string
DynamicAlgoBase::to_filename(const string s)
{
  string result;
  for (unsigned int i = 0; i < s.size(); i++)
  {
    if (isalnum(s[i]))
    {
      result += s[i];
    }
  }
  return result;
}


DynamicLoader::DynamicLoader() :
  map_crowd_("DynamicLoader: One compilation at a time."),
  compilation_cond_("DynamicLoader: waits for compilation to finish."),
  map_lock_("DynamicLoader: controls mutable access to the map.")
{
  map_lock_.lock();
  algo_map_.clear();
  map_lock_.unlock();
}

DynamicLoader::~DynamicLoader() 
{
  map_lock_.lock();
  algo_map_.clear();
  map_lock_.unlock();
}


//! DynamicLoader::scirun_loader
//! 
//! How to get at the global loader for scirun.
DynamicLoader& 
DynamicLoader::scirun_loader() {
  if (scirun_loader_ == 0) {
    scirun_loader_lock_.lock();
    if (scirun_loader_ == 0) {
      scirun_loader_ = new DynamicLoader;
    }
    scirun_loader_lock_.unlock();
  }
  return *scirun_loader_;
}

//! DynamicLoader::entry_exists
//! 
//! Convenience function to query the map, but thread safe.
bool 
DynamicLoader::entry_exists(const string &entry)
{
  map_lock_.lock();
  bool rval = algo_map_.find(entry) != algo_map_.end();
  map_lock_.unlock();
  return rval;
}

//! DynamicLoader::entry_is_null
//! 
//! Convenience function to query the value in the map, but thread safe.
bool 
DynamicLoader::entry_is_null(const string &entry)
{
  map_lock_.lock();
  bool rval =  (algo_map_.find(entry) != algo_map_.end() && 
		algo_map_[entry] == 0);
  map_lock_.unlock();
  return rval;
}

//! DynamicLoader::wait_for_current_compile
//! 
//! Block if the lib associated with entry is compiling now.
//! The only way false is returned, is for the compile to fail.
bool
DynamicLoader::wait_for_current_compile(const string &entry)
{
  while (entry_is_null(entry)) {
    // another thread is compiling this lib, so wait.
    map_lock_.lock();
    compilation_cond_.wait(map_lock_);
    map_lock_.unlock();
  }
  // if the map entry no longer exists, compilation failed.
  if (! entry_exists(entry)) return false;
  // The maker fun has been stored by another thread.
  ASSERT(! entry_is_null(entry));
  return true;
}

//! DynamicLoader::compile_and_store
//! 
//! Compile and store the maker function mapped to the lib name.
//! The sychronization code allows multiple threads to compile different
//! libs at the same time, but forces only one thread can compile any one
//! lib.
bool
DynamicLoader::compile_and_store(const CompileInfo &info, bool maybe_compile_p)
{  
  bool do_compile = false;
  
  if (! entry_exists(info.filename_)) {
    // first attempt at creation of this lib
    map_crowd_.writeLock();
    if (! entry_exists(info.filename_)) {
      // create an empty entry, to catch threads chasing this one.
      map_lock_.lock();
      algo_map_[info.filename_] = 0; 
      map_lock_.unlock();
      do_compile = true; // this thread is compiling.
    } 
    map_crowd_.writeUnlock();
  }

  if (! do_compile) {
    if (! wait_for_current_compile(info.filename_)) return false;
  }

  // Try to load a .so that is already compiled
  string full_so = OTF_OBJ_DIR + string("/") + 
    info.filename_ + string("so");

  LIBRARY_HANDLE so = 0;
  struct stat buf;
  if (stat(full_so.c_str(), &buf) == 0) {
    compile_so(info.filename_); // make sure
    so = GetLibraryHandle(full_so.c_str());
  } else {
    // the lib does not exist.  
    create_cc(info);
    compile_so(info.filename_);
    so = GetLibraryHandle(full_so.c_str());

    if (maybe_compile_p && so == 0)
    {
      create_empty_cc(info);
      compile_so(info.filename_);
      so = GetLibraryHandle(full_so.c_str());
    }
     
    if (so == 0) { // does not compile
      cerr << "does not compile" << endl;
      cerr << "DYNAMIC COMPILATION ERROR: " << full_so 
	   << " does not compile!!" << endl;
      cerr << SOError() << endl;
      // Remove the null ref for this lib from the map.
      map_lock_.lock();
      algo_map_.erase(info.filename_);
      map_lock_.unlock();
      // wake up all sleepers.
      compilation_cond_.conditionBroadcast();
      return false;
    }
  }

  maker_fun maker = 0;
  maker = (maker_fun)GetHandleSymbolAddress(so, "maker");
  
  if (maker == 0) {
    cerr << "DYNAMIC LIB ERROR: " << full_so 
	 << " no maker function!!" << endl;
    cerr << SOError() << endl;
    // Remove the null ref for this lib from the map.
    map_lock_.lock();
    algo_map_.erase(info.filename_);
    map_lock_.unlock();
    // wake up all sleepers.
    compilation_cond_.conditionBroadcast();
    return false;
  }
  // store this so that we can get at it again.
  store(info.filename_, maker);
  // wake up all sleepers. 
  compilation_cond_.conditionBroadcast();
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

  cerr << "Executing: " << command << endl;
  bool compiled =  sci_system(command.c_str()) == 0; 
  if(!compiled) {
    cerr << "DynamicLoader::compile_so() error: "
	 << "system call failed:" << endl << command << endl;
  } else {
    cerr << "DynamicLoader - successfully compiled " << file + "so" << endl;
  }

  return compiled;
}



//! DynamicLoader::create_cc
//!
//! Write a .cc file, from the compile info.
bool 
DynamicLoader::create_cc(const CompileInfo &info)
{
  const string STD_STR("std::");

  // Try to open the file for writing.
  string full = OTF_OBJ_DIR + "/" + info.filename_ + "cc";
  ofstream fstr(full.c_str());

  if (!fstr) {
    cerr << "DynamicLoader::create_cc could not create file " << full << endl;
    return false;
  }
  fstr << "// This is an autamatically generated file, do not edit!" << endl;

  // generate standard includes
  list<string>::const_iterator iter = info.includes_.begin();
  while (iter != info.includes_.end()) { 
    const string &s = *iter;
    if (s.substr(0, 5) == STD_STR)
    {
      string std_include = s.substr(5, s.length() -1);
      fstr << "#include <" << std_include << ">" << endl;
    }
    ++iter;
  }

  // generate other includes
  iter = info.includes_.begin();
  while (iter != info.includes_.end()) { 
    const string &s = *iter;
    if (!((s.substr(0, 5) == STD_STR) || s == "builtin"))
    {
      fstr << "#include \"" << s << "\"" << endl;
    }
    ++iter;
  }

  // output namespaces
  CompileInfo::ci_map_type::const_iterator nsiter = info.namespaces_.begin();
  while (nsiter != info.namespaces_.end()) { 
    const string &s = (*nsiter).first;
    if (s != "builtin") {
      fstr << "using namespace " << s << ";" << endl;
    }
    ++nsiter;
  }


  fstr << endl;

  fstr << "extern \"C\" {"  << endl
       << info.base_class_name_ << "* maker() {" << endl
       << "  return scinew "<< info.template_class_name_ << "<" 
       << info.template_arg_ << ">;" << endl
       << "}" << endl << "}" << endl;

  cerr << "DynamicLoader - successfully created " << full << endl;
  return true;
}


//! DynamicLoader::create_empty_cc
//!
//! Write a .cc file, from the compile info.
//! It contains an empty maker function.  Used if the actual compilation
//! fails.
bool 
DynamicLoader::create_empty_cc(const CompileInfo &info)
{
  const string STD_STR("std::");

  // Try to open the file for writing.
  string full = OTF_OBJ_DIR + "/" + info.filename_ + "cc";
  ofstream fstr(full.c_str());

  if (!fstr) {
    cerr << "DynamicLoader::create_cc could not create file " << full << endl;
    return false;
  }
  fstr << "// This is an autamatically generated file, do not edit!" << endl;

  // generate standard includes
  list<string>::const_iterator iter = info.includes_.begin();
  while (iter != info.includes_.end()) { 
    const string &s = *iter;
    if (s.substr(0, 5) == STD_STR)
    {
      string std_include = s.substr(5, s.length() -1);
      fstr << "#include <" << std_include << ">" << endl;
    }
    ++iter;
  }

  // generate other includes
  iter = info.includes_.begin();
  while (iter != info.includes_.end()) { 
    const string &s = *iter;
    if (!((s.substr(0, 5) == STD_STR) || s == "builtin"))
    {
      fstr << "#include \"" << s << "\"" << endl;
    }
    ++iter;
  }

  // output namespaces
  CompileInfo::ci_map_type::const_iterator nsiter = info.namespaces_.begin();
  while (nsiter != info.namespaces_.end()) { 
    const string &s = (*nsiter).first;
    if (s != "builtin") {
      fstr << "using namespace " << s << ";" << endl;
    }
    ++nsiter;
  }


  fstr << endl;

  fstr << "extern \"C\" {"  << endl
       << info.base_class_name_ << "* maker() {" << endl
       << "  return 0;" << endl
       << "}" << endl << "}" << endl;

  cerr << "DynamicLoader - successfully created " << full << endl;
  return true;
}


void 
DynamicLoader::store(const string &name, maker_fun m)
{
  map_lock_.lock();
  algo_map_[name] = m;
  map_lock_.unlock();
}

bool 
DynamicLoader::fetch(const CompileInfo &ci, DynamicAlgoHandle &algo)
{
  bool rval = false;
  // block in case we get here while it is compiling.
  if (! wait_for_current_compile(ci.filename_)) return false;

  map_crowd_.readLock();
  map_lock_.lock();
  map_type::iterator loc = algo_map_.find(ci.filename_);
  if (loc != algo_map_.end()) {
    maker_fun m = loc->second;
    algo = DynamicAlgoHandle(m());
    rval = true;
  }
  map_lock_.unlock();
  map_crowd_.readUnlock();
  return rval;
}

bool 
DynamicLoader::get(const CompileInfo &ci, DynamicAlgoHandle &algo)
{
  return (fetch(ci, algo) ||
	  (compile_and_store(ci, false) && fetch(ci, algo)));
}


bool
DynamicLoader::maybe_get(const CompileInfo &ci, DynamicAlgoHandle &algo)
{
  return (fetch(ci, algo) ||
	  (compile_and_store(ci, true) && fetch(ci, algo)));
}

} // End namespace SCIRun


