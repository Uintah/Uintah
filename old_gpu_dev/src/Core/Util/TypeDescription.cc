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

//    File   : TypeDescription.cc
//    Author : Martin Cole
//    Date   : Mon May 14 10:20:21 2001

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/Mutex.h>
#include <map>
#include <iostream>

using namespace std;
namespace SCIRun {

struct KillMap {
  KillMap();
  ~KillMap();
};

KillMap::KillMap()
{
}

static map<string, const TypeDescription*>* types = 0;
static vector<const TypeDescription*>* typelist=0;
static Mutex typelist_lock("TypeDescription::typelist lock");
static bool killed=false;

KillMap::~KillMap()
{
  if(!types){
    ASSERT(!killed);
    ASSERT(!typelist);
    return;
  }
  killed=true;
  vector<const TypeDescription*>::iterator iter = typelist->begin();
  for(;iter != typelist->end();iter++)
    delete *iter;
  delete types;
  delete typelist;
}

KillMap killit;


void
TypeDescription::register_type()
{
  typelist_lock.lock();
  if (!types)
  {
    ASSERT(!killed);
    ASSERT(!typelist);

    // This will make sure that if types was not initialized when we
    // entered this block, that we will not try and reinitialize types
    // and typelist.
    if (!types)
    {
      types = scinew map<string, const TypeDescription*>;
      typelist = scinew vector<const TypeDescription*>;
    }
  }

  map<string, const TypeDescription*>::iterator iter = types->find(get_name());
  if (iter == types->end())
  {
    (*types)[get_name()] = this;
  }
  typelist->push_back(this);
  
  typelist_lock.unlock();
}


TypeDescription::TypeDescription(const string &name, 
				 const string &path,
				 const string &namesp, 
				 category_e c) : 
  subtype_(0), 
  name_(name),
  h_file_path_(path),
  namespace_(namesp),
  category_(c)
{
  register_type();
}

TypeDescription::TypeDescription(const string &name, 
				 td_vec* sub, 
				 const string &path,
				 const string &namesp,
				 category_e c) : 
  subtype_(sub),
  name_(name),
  h_file_path_(path),
  namespace_(namesp),
  category_(c)
{
  register_type();
}

TypeDescription::~TypeDescription()
{
  if (subtype_) delete subtype_;
}

string 
TypeDescription::get_name( const string & type_sep_start /* = "<"  */, 
			   const string & type_sep_end   /* = "> " */ ) const
{
  const string comma(",");
  bool do_end = true;
  if(subtype_) {
    string rval = name_ + type_sep_start;
    td_vec::iterator iter = subtype_->begin();
    while (iter != subtype_->end()) {
      rval+=(*iter)->get_name(type_sep_start, type_sep_end);
      ++iter;
      if (iter != subtype_->end()) {
	if (type_sep_start == type_sep_end) {
	  do_end = false;
	} else {
	  rval += comma;
	}
      }
    }
    if (do_end) rval += type_sep_end;
    return rval;
  } else {
    return name_;
  }
}

// substitute one of the subtype names with the name provided.
string 
TypeDescription::get_similar_name(const string &substitute,
				  const int pos,
				  const string &type_sep_start, 
				  const string &type_sep_end) const 
{
  const string comma(",");
  bool do_end = true;
  if(subtype_) {
    string rval = name_ + type_sep_start;
    td_vec::iterator iter = subtype_->begin();
    int count = 0;
    while (iter != subtype_->end()) {
      if (pos == count) {
	rval += substitute;
      } else {
	rval+=(*iter)->get_name(type_sep_start, type_sep_end);
      }
      ++iter;
      if (iter != subtype_->end()) {
	if (type_sep_start == type_sep_end) {
	  do_end = false;
	} else {
	  rval += comma;
	}
      }
      ++count;
    }
    if (do_end) rval += type_sep_end;
    return rval;
  } else {
    return name_;
  }
}


string 
TypeDescription::get_filename() const
{
  string s = get_name();
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


#if !defined( REDSTORM )
void 
TypeDescription::fill_compile_info(CompileInfo *ci) const
{
  switch (category_) {
  case DATA_E:
    ci->add_data_include(get_h_file_path());
    break;
  case BASIS_E:
    ci->add_basis_include(get_h_file_path());
    break;
  case MESH_E:
    ci->add_mesh_include(get_h_file_path());
    break;
  case CONTAINER_E:
    ci->add_container_include(get_h_file_path());
    break;
  case FIELD_E:
    ci->add_field_include(get_h_file_path());
    break;
  default:
    ci->add_include(get_h_file_path());
  }


  ci->add_namespace(get_namespace());
  if(subtype_) {
    td_vec::iterator iter = subtype_->begin();
    while (iter != subtype_->end()) {
      (*iter)->fill_compile_info(ci);
      ++iter;
    }
  }
}
#endif

const TypeDescription* 
TypeDescription::lookup_type(const std::string& t)
{
  if(!types) {
    typelist_lock.lock();
    if (!types) {
      types=scinew map<string, const TypeDescription*>;
      typelist=new vector<const TypeDescription*>;
    }
    typelist_lock.unlock();
  }
  
  map<string, const TypeDescription*>::iterator iter = types->find(t);
   if(iter == types->end())
      return 0;
   return iter->second;
}

string TypeDescription::cc_to_h(const string &dot_cc)
{
  const unsigned int len = dot_cc.length();
  string dot_h;
  if (len > 3 && dot_cc.substr(len-3, len) == ".cc") {
    dot_h = dot_cc.substr(0, len-3) + ".h";
  } else {
    cerr << "Warning: TypeDescription::cc_to_h input does not end in .cc" 
	 << endl << "the string: '" << dot_cc << "'" << endl;
    dot_h = dot_cc;
  }
  return dot_h;
}


TypeDescription::Register::Register(const TypeDescription* /* td*/)
{
  // Registration happens in CTOR
}

TypeDescription::Register::~Register()
{
}

const TypeDescription* get_type_description(double*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("double", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(long*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("long", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(float*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("float", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(short*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("short", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(unsigned short*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("unsigned short", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(int*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("int", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(unsigned int*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("unsigned int", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(char*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("char", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(unsigned char*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("unsigned char", "builtin", "builtin");
  }
  return td;
}

const TypeDescription* get_type_description(string*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("string", "std::string", "std");
  }
  return td;
}

const TypeDescription* get_type_description(unsigned long*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("unsigned long", "builtin", "builtin");
  }
  return td;
}

} // end namespace SCIRun
