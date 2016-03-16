/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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
//    File   : TypeDescription.h
//    Author : Martin Cole
//    Date   : Mon May 14 10:16:28 2001

#if ! defined(Disclosure_TypeDescription_h)
#define Disclosure_TypeDescription_h

#include <Core/Malloc/Allocator.h>

#include <vector>
#include <string>

namespace SCIRun {

class STypeDescription {
public:
  enum category_e {
    DATA_E,
    BASIS_E,
    MESH_E,
    CONTAINER_E,
    FIELD_E,
    OTHER_E
  }; 

  typedef std::vector<const STypeDescription*> td_vec;

  STypeDescription(const std::string& name,
		  category_e c = OTHER_E);
  STypeDescription(const std::string& name,
		  td_vec *sub, // this takes ownership of the memory. 
		  category_e c = OTHER_E);
  ~STypeDescription();
     
  td_vec* get_sub_type() const {
    return subtype_;
  }
  //! The arguments determine how the templated types are separated.
  //! default is "<" and "> "
  std::string get_name(const std::string & type_sep_start = "<",
		  const std::string & type_sep_end = "> ") const;
  std::string get_similar_name(const std::string &substitute,
			  const int pos,
			  const std::string & type_sep_start = "<",
			  const std::string & type_sep_end = "> ") const;

  std::string get_filename() const;


  struct Register {
    Register(const STypeDescription*);
    ~Register();
  };



  static const STypeDescription* lookup_type(const std::string&);

private:
  td_vec                     *subtype_;
  std::string                     name_;
  category_e                 category_;
  // Hide these methods
  STypeDescription(const STypeDescription&);
  STypeDescription& operator=(const STypeDescription&);

  void register_type();
};


const STypeDescription* get_type_description(double*);
const STypeDescription* get_type_description(long*);
const STypeDescription* get_type_description(float*);
const STypeDescription* get_type_description(short*);
const STypeDescription* get_type_description(unsigned short*);
const STypeDescription* get_type_description(int*);
const STypeDescription* get_type_description(unsigned int*);
const STypeDescription* get_type_description(char*);
const STypeDescription* get_type_description(unsigned char*);
const STypeDescription* get_type_description(bool*);
const STypeDescription* get_type_description(std::string*);
const STypeDescription* get_type_description(unsigned long*);

template <class T>
const STypeDescription* get_type_description(std::vector<T>*)
{
  static STypeDescription* td = 0;
  if(!td){
    const STypeDescription *sub = SCIRun::get_type_description((T*)0);
    STypeDescription::td_vec *subs = scinew STypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew STypeDescription("vector", subs, STypeDescription::CONTAINER_E);
  }
  return td;
}

template <class T1, class T2>
const STypeDescription* get_type_description (std::pair<T1,T2> *)
{
  static STypeDescription* td = 0;
  if(!td){
    const STypeDescription *sub1 = SCIRun::get_type_description((T1*)0);
    const STypeDescription *sub2 = SCIRun::get_type_description((T2*)0);
    STypeDescription::td_vec *subs = scinew STypeDescription::td_vec(2);
    (*subs)[0] = sub1;
    (*subs)[1] = sub2;
    td = scinew STypeDescription("pair", subs, STypeDescription::CONTAINER_E);
  }
  return td;

}

} // End namespace SCIRun

#endif //Disclosure_STypeDescription_h

