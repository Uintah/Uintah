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

#if ! defined(Disclosure_TypeDescription_h)
#define Disclosure_TypeDescription_h

#include <Core/Malloc/Allocator.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;
using std::vector;
using std::pair;

class CompileInfo;

class TypeDescription {
public:
  typedef vector<const TypeDescription*> td_vec;

  TypeDescription(const string& name,
		  const string& path,
		  const string& namesp);
  TypeDescription(const string& name, 
		  td_vec *sub, // this takes ownership of the memory. 
		  const string& path,
		  const string& namesp);
  ~TypeDescription();
     
  td_vec* get_sub_type() const {
    return subtype_;
  }
  //! The arguments determine how the templated types are separated.
  //! default is "<" and "> "
  string get_name(string type_sep_start = "<", 
		  string type_sep_end = "> ") const;

  string get_filename() const;

  string get_h_file_path() const { return h_file_path_; }
  string get_namespace() const { return namespace_; }

  struct Register {
    Register(const TypeDescription*);
    ~Register();
  };

  void fill_compile_info(CompileInfo *ci) const;

  //! convert a string that ends in .cc to end in .h
  static string cc_to_h(const string &dot_cc);

  static const TypeDescription* lookup_type(const string&);

private:
  td_vec                     *subtype_;
  string                     name_;
  string                     h_file_path_;
  string                     namespace_;

  // Hide these methods
  TypeDescription(const TypeDescription&);
  TypeDescription& operator=(const TypeDescription&);

  void register_type();
};


const TypeDescription* get_type_description(double*);
const TypeDescription* get_type_description(long*);
const TypeDescription* get_type_description(float*);
const TypeDescription* get_type_description(short*);
const TypeDescription* get_type_description(unsigned short*); 
const TypeDescription* get_type_description(int*);
const TypeDescription* get_type_description(unsigned int*);
const TypeDescription* get_type_description(char*);
const TypeDescription* get_type_description(unsigned char*);
const TypeDescription* get_type_description(bool*);
const TypeDescription* get_type_description(string*);
const TypeDescription* get_type_description(unsigned long*);

template <class T>
const TypeDescription* get_type_description(vector<T>*)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("vector", subs, "std::vector", "std");
  }
  return td;
}

template <class T1, class T2>
const TypeDescription* get_type_description (pair<T1,T2> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub1 = SCIRun::get_type_description((T1*)0);
    const TypeDescription *sub2 = SCIRun::get_type_description((T2*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(2);
    (*subs)[0] = sub1;
    (*subs)[1] = sub2;
    td = scinew TypeDescription("pair", subs, "std::utility", "std");
  }
  return td;

}

} // End namespace SCIRun

#endif //Disclosure_TypeDescription_h

