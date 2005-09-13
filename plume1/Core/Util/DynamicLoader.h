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
#include <Core/Util/ProgressReporter.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <list>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

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
  void add_post_include(const string &post);
  void create_cc(ostream &fstr, bool empty) const;
  
  string             filename_;
  list<string>       includes_;
  ci_map_type        namespaces_;
  string             base_class_name_;
  string             template_class_name_;
  string             template_arg_;
  string             post_include_extra_;

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
  bool compile_and_store(const CompileInfo &info, bool maybe_compile,
			 ProgressReporter *pr);
  void cleanup_failed_compile(CompileInfoHandle info);

  //! All modules should use this function to get the loader.
  static DynamicLoader& scirun_loader();

private:
  bool create_cc(const CompileInfo &info, bool empty, ProgressReporter *pr);
  bool compile_so(const CompileInfo &info, ProgressReporter *pr);
  void store( const string &, maker_fun);
  bool entry_exists(const string &entry);
  bool entry_is_null(const string &entry);
  bool wait_for_current_compile(const string &entry);
  
  typedef map<string, maker_fun> map_type;
  map_type              algo_map_;
  
  //! Thread Safety. 
  CrowdMonitor          map_crowd_;
  ConditionVariable     compilation_cond_;
  Mutex                 map_lock_;
  static string		otf_dir();

  //! static vars.
  static DynamicLoader *scirun_loader_;
  static Mutex          scirun_loader_init_lock_;
  
};

} // End namespace SCIRun

#endif //Disclosure_DynamicLoader_h
