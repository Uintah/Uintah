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
//    File   : DynamicLoader.h
//    Author : Martin Cole
//    Date   : Mon May 21 10:57:54 2001

#if ! defined(Disclosure_DynamicLoader_h)
#define Disclosure_DynamicLoader_h

#include <Core/Containers/Handle.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>

#include <map>
#include <list>
#include <string>
#include <iostream>

namespace SCIRun {
using namespace std;

struct CompileInfo
{
public:
  typedef map<string, int> ci_map_type; //! unique keys.
  //! construct with filename, base class name, and template class name.
  CompileInfo(const string &fn, const string &bcn, 
	      const string &tcn, const string &tcdec);
  
  //! add to the list of files to include.
  void add_include(const string &inc);
  void add_namespace(const string &inc) { namespaces_[inc] = 1; }
  
  string             filename_;
  list<string>       includes_;
  ci_map_type        namespaces_;
  string             base_class_name_;
  string             template_class_name_;
  string             template_arg_;

  int       ref_cnt;
};

typedef Handle<CompileInfo> CompileInfoHandle;



//! A type that maker functions can create, and DynamicLoader can store.
//! All algorithms that support the dynamic loading concept must 
//! inherit from this.
struct DynamicAlgoBase { 
  int       ref_cnt;
  Mutex     lock;

  DynamicAlgoBase();
  virtual ~DynamicAlgoBase();

  static string to_filename(const string s);
};

typedef LockingHandle<DynamicAlgoBase> DynamicAlgoHandle;

class DynamicLoader
{
public:
  typedef DynamicAlgoBase* (*maker_fun)();

  DynamicLoader();
  ~DynamicLoader();

  //! Compile and load .so for the selected manipulation
  bool get(const CompileInfo &info, DynamicAlgoHandle&);
  bool maybe_get(const CompileInfo &info, DynamicAlgoHandle&);
  bool fetch(const CompileInfo &info, DynamicAlgoHandle&);
  bool compile_and_store(const CompileInfo &info, bool maybe_compile = false,
			 ostream &serr = std::cerr);
  void remove_cc(const CompileInfo &info, ostream &serr);

  //! All modules should use this function to get the loader.
  static DynamicLoader& scirun_loader();

private:
  bool create_cc(const CompileInfo &info, bool empty, ostream &serr);
  bool compile_so(const CompileInfo &info, ostream &serr);
  void store( const string &, maker_fun);
  bool entry_exists(const string &entry);
  bool entry_is_null(const string &entry);
  bool wait_for_current_compile(const string &entry);
  
  bool copy_makefile_to(const string &dir);
  const string &get_compile_dir();
  bool validate_compile_dir(string &dir);
  
  typedef map<string, maker_fun> map_type;
  map_type              algo_map_;
  
  //! Thread Safety. 
  CrowdMonitor          map_crowd_;
  ConditionVariable     compilation_cond_;
  Mutex                 map_lock_;

  //! static vars.
  static DynamicLoader *scirun_loader_;
  static Mutex          scirun_loader_init_lock_;
  static string		otf_dir_;
  static bool		otf_dir_found_;
};

} // End namespace SCIRun

#endif //Disclosure_DynamicLoader_h
