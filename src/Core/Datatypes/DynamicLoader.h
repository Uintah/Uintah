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

#if ! defined(Datatypes_DynamicLoader_h)
#define Datatypes_DynamicLoader_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>

#include <map>
#include <string>
#include <vector>

namespace SCIRun {
using namespace std;

struct CompileInfo
{
public:
  //! construct with filename, base class name, and template class name.
  CompileInfo(const string &fn, const string &bcn, 
	      const string &tcn, const string &tcdec);
  
  //! add to the list of files to include.
  void add_include(const string &inc) { includes_.push_back(inc); }
  
  string             filename_;
  vector<string>     includes_;
  string             base_class_name_;
  string             template_class_name_;
  string             template_arg_;
};

//! A type that maker functions can create, and DynamicLoader can store.
//! All algorithms that support the dynamic loading concept must 
//! inherit from this.
struct DynamicAlgoBase : public Datatype { // inherit from Datatype to get 
  virtual ~DynamicAlgoBase() {}            // handle functionality. 
  virtual void io(Piostream &) {}          // no Pio for algorithms
};

typedef LockingHandle<DynamicAlgoBase> DynamicAlgoHandle;

class DynamicLoader
{
public:
  DynamicLoader();
  ~DynamicLoader();

  // Compile and load .so for the selected manipulation
  bool compile_and_store(const CompileInfo &info);
  bool get( const string &, DynamicAlgoHandle&);


private:
  bool create_cc(const CompileInfo &info);
  bool compile_so(const string &file);
  void store( const string &, DynamicAlgoHandle);

  typedef map<string, DynamicAlgoHandle> map_type;
  map_type algo_map_;
};

} // End namespace SCIRun

#endif //Datatypes_DynamicLoader_h
