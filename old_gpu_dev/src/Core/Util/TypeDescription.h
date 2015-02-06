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

//    File   : TypeDescription.h
//    Author : Martin Cole
//    Date   : Mon May 14 10:16:28 2001

#if ! defined(Disclosure_TypeDescription_h)
#define Disclosure_TypeDescription_h

#include <Core/Malloc/Allocator.h>

#include <vector>
#include <string>

#include <Core/Util/share.h>

namespace SCIRun {

using std::string;
using std::vector;
using std::pair;

struct CompileInfo;

class SCISHARE TypeDescription {
public:
  enum category_e {
    DATA_E,
    BASIS_E,
    MESH_E,
    CONTAINER_E,
    FIELD_E,
    OTHER_E
  }; 

  typedef vector<const TypeDescription*> td_vec;

  TypeDescription(const string& name,
		  const string& path,
		  const string& namesp,
		  category_e c = OTHER_E);
  TypeDescription(const string& name, 
		  td_vec *sub, // this takes ownership of the memory. 
		  const string& path,
		  const string& namesp,
		  category_e c = OTHER_E);
  ~TypeDescription();
     
  td_vec* get_sub_type() const {
    return subtype_;
  }
  //! The arguments determine how the templated types are separated.
  //! default is "<" and "> "
  string get_name(const string & type_sep_start = "<", 
		  const string & type_sep_end = "> ") const;
  string get_similar_name(const string &substitute,
			  const int pos,
			  const string & type_sep_start = "<", 
			  const string & type_sep_end = "> ") const;

  string get_filename() const;

  string get_h_file_path() const { return h_file_path_; }
  string get_namespace() const { return namespace_; }

  struct Register {
    Register(const TypeDescription*);
    ~Register();
  };

#if !defined( REDSTORM )
  void fill_compile_info(CompileInfo *ci) const;
#endif

  //! convert a string that ends in .cc to end in .h
  static string cc_to_h(const string &dot_cc);

  static const TypeDescription* lookup_type(const string&);

private:
  td_vec                     *subtype_;
  string                     name_;
  string                     h_file_path_;
  string                     namespace_;
  category_e                 category_;
  // Hide these methods
  TypeDescription(const TypeDescription&);
  TypeDescription& operator=(const TypeDescription&);

  void register_type();
};


SCISHARE const TypeDescription* get_type_description(double*);
SCISHARE const TypeDescription* get_type_description(long*);
SCISHARE const TypeDescription* get_type_description(float*);
SCISHARE const TypeDescription* get_type_description(short*);
SCISHARE const TypeDescription* get_type_description(unsigned short*); 
SCISHARE const TypeDescription* get_type_description(int*);
SCISHARE const TypeDescription* get_type_description(unsigned int*);
SCISHARE const TypeDescription* get_type_description(char*);
SCISHARE const TypeDescription* get_type_description(unsigned char*);
SCISHARE const TypeDescription* get_type_description(bool*);
SCISHARE const TypeDescription* get_type_description(string*);
SCISHARE const TypeDescription* get_type_description(unsigned long*);

template <class T>
const TypeDescription* get_type_description(vector<T>*)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("vector", subs, "std::vector", "std",
				TypeDescription::CONTAINER_E);
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
    td = scinew TypeDescription("pair", subs, "std::utility", "std",
				TypeDescription::CONTAINER_E);
  }
  return td;

}

} // End namespace SCIRun

#endif //Disclosure_TypeDescription_h

