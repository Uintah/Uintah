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
//    File   : TypeDescription.h
//    Author : Martin Cole
//    Date   : Mon May 14 10:16:28 2001

#if ! defined(Datatypes_TypeDescription_h)
#define Datatypes_TypeDescription_h

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Tensor.h>
#include <vector>
#include <string>


using namespace std;

namespace SCIRun {

class CompileInfo;

class TypeDescription {
public:
  TypeDescription(const string& name,
		  const string& path);
  TypeDescription(const string& name, const TypeDescription *td, 
		  const string& path);
  ~TypeDescription();
     
  const TypeDescription* get_sub_type() const {
    return subtype_;
  }
  //! The arguments determine how the templated types are separated.
  //! default is "<" and "> "
  string get_name(string type_sep_start = "<", 
		  string type_sep_end = "> ") const;

  string get_h_file_path() const { return h_file_path_; }

  struct Register {
    Register(const TypeDescription*);
    ~Register();
  };

  void fill_includes(CompileInfo *ci) const;
  
  static const TypeDescription* lookup_type(const string&);

private:

  //FIX_ME need to support multiple subtypes....foo<bar, foobar, bilbo>
  const TypeDescription*     subtype_;
  string                     name_;
  string                     h_file_path_;
       
  // Hide these methods
  TypeDescription(const TypeDescription&);
  TypeDescription& operator=(const TypeDescription&);

  void register_type();
};


const TypeDescription* get_type_description(double*);
const TypeDescription* get_type_description(float*);
const TypeDescription* get_type_description(short*);
const TypeDescription* get_type_description(int*);
const TypeDescription* get_type_description(unsigned char*);
const TypeDescription* get_type_description(bool*);
const TypeDescription* get_type_description(Vector*);
const TypeDescription* get_type_description(Tensor*);
const TypeDescription* get_type_description(Point*);
const TypeDescription* get_type_description(Transform*);
const TypeDescription* get_type_description(string*);

template <class T>
const TypeDescription* get_type_description(vector<T>*)
{
  static TypeDescription* td = 0;
  static string v("vector");
  static string path("std::vector"); // dynamic loader will parse off the std
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    td = scinew TypeDescription(v, sub, path);
  }
  return td;
}

} // End namespace SCIRun

#endif //Datatypes_TypeDescription_h

