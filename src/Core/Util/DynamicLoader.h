/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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

#include <map>
#include <list>
#include <string>
#include <iostream>

#include <Core/Util/share.h>

namespace SCIRun {

struct SCISHARE CompileInfo
{
public:
  typedef std::map<std::string, int> ci_map_type; //! unique keys.
  //! construct with filename, base class name, and template class name.
  CompileInfo(const std::string &fn, const std::string &bcn, 
	      const std::string &tcn, const std::string &tcdec);
  
  //! add to the list of files to include.
  void add_include(const std::string &inc);
  void add_data_include(const std::string &inc);
  void add_basis_include(const std::string &inc);
  void add_mesh_include(const std::string &inc);
  void add_container_include(const std::string &inc);
  void add_field_include(const std::string &inc);
  void add_namespace(const std::string &inc) { namespaces_[inc] = 1; }
  void add_pre_include(const std::string &pre);
  void add_post_include(const std::string &post);
  void create_cc(std::ostream &fstr, bool empty) const;
  
  std::string             filename_;
  std::list<std::string>  includes_;
  std::list<std::string>  data_includes_;
  std::list<std::string>  basis_includes_;
  std::list<std::string>  mesh_includes_;
  std::list<std::string>  container_includes_;
  std::list<std::string>  field_includes_;
  ci_map_type             namespaces_;
  std::string             base_class_name_;
  std::string             template_class_name_;
  std::string             template_arg_;
  std::string             pre_include_extra_;
  std::string             post_include_extra_;

  //! For regression testing after the function has been compiled remove
  //! library if it was not there, we are just testing the regression.
  //! This prevents over crowding of the on the fly libs directory.
  bool               keep_library_;

  int       ref_cnt;
};

typedef Handle<CompileInfo> CompileInfoHandle;



//! A type that maker functions can create, and DynamicLoader can store.
//! All algorithms that support the dynamic loading concept must 
//! inherit from this.
struct SCISHARE DynamicAlgoBase { 
  int       ref_cnt;
  Mutex     lock;

  DynamicAlgoBase();
  virtual ~DynamicAlgoBase();

  static std::string to_filename(const std::string s);
};

typedef LockingHandle<DynamicAlgoBase> DynamicAlgoHandle;

class SCISHARE DynamicLoader
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
  void cleanup_compile(CompileInfoHandle info);

  //! All modules should use this function to get the loader.
  static DynamicLoader& scirun_loader();

private:
  bool create_cc(const CompileInfo &info, bool empty, ProgressReporter *pr);
  bool compile_so(const CompileInfo &info, ProgressReporter *pr);
  void store( const std::string &, maker_fun);
  bool entry_exists(const std::string &entry);
  bool entry_is_null(const std::string &entry);
  bool wait_for_current_compile(const std::string &entry);
  
  typedef std::map<std::string, maker_fun> map_type;
  map_type              algo_map_;
  
  //! Thread Safety. 
  CrowdMonitor          map_crowd_;
  ConditionVariable     compilation_cond_;
  Mutex                 map_lock_;
  static std::string	otf_dir();

  //! static vars.
  static DynamicLoader *scirun_loader_;
  static Mutex          scirun_loader_init_lock_;
  
};

} // End namespace SCIRun

#endif //Disclosure_DynamicLoader_h
