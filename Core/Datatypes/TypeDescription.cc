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
//    File   : TypeDescription.cc
//    Author : Martin Cole
//    Date   : Mon May 14 10:20:21 2001

#include <Core/Datatypes/TypeDescription.h>
#include <Core/Datatypes/DynamicLoader.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <sci_defs.h>
#include <map>
#include <vector>
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

void TypeDescription::register_type()
{
  if(!types){
    ASSERT(!killed);
    ASSERT(!typelist)
    types=scinew map<string, const TypeDescription*>;
    typelist=new vector<const TypeDescription*>;
  }
  map<string, const TypeDescription*>::iterator iter = types->find(get_name());
  if(iter == types->end())
    (*types)[get_name()]=this;
  typelist->push_back(this);
}

TypeDescription::TypeDescription(const string& name, const string& path) : 
  subtype_(0), 
  name_(name),
  h_file_path_(path)
{
  register_type();
}

TypeDescription::TypeDescription(const string& name, 
				 td_vec* sub, 
				 const string& path) : 
  subtype_(sub),
  name_(name),
  h_file_path_(path)
{
  register_type();
}

TypeDescription::~TypeDescription()
{
  if (subtype_) delete subtype_;
}

string 
TypeDescription::get_name(string type_sep_start, 
			  string type_sep_end) const
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

void 
TypeDescription::fill_includes(CompileInfo *ci) const
{
  ci->add_include(get_h_file_path());
  if(subtype_) {
    td_vec::iterator iter = subtype_->begin();
    while (iter != subtype_->end()) {
      (*iter)->fill_includes(ci);
      ++iter;
    }
  }
}


const TypeDescription* 
TypeDescription::lookup_type(const std::string& t)
{
  if(!types)
    types=scinew map<string, const TypeDescription*>;   
  
  map<string, const TypeDescription*>::iterator iter = types->find(t);
   if(iter == types->end())
      return 0;
   return iter->second;
}

TypeDescription::Register::Register(const TypeDescription*/* td*/)
{
  // Registration happens in CTOR
}

TypeDescription::Register::~Register()
{
}

const TypeDescription* get_type_description(double*)
{
  static TypeDescription* td = 0;
  static string nm("double");
  static string path("builtin");
  if(!td){
    td = scinew TypeDescription(nm, path);
  }
  return td;
}

const TypeDescription* get_type_description(long*)
{
  static TypeDescription* td = 0;
  static string nm("long");
  static string path("builtin");
  if(!td){
    td = scinew TypeDescription(nm, path);
  }
  return td;
}

const TypeDescription* get_type_description(float*)
{
  static TypeDescription* td = 0;
  static string nm("float");
  static string path("builtin");
  if(!td){
    td = scinew TypeDescription(nm, path);
  }
  return td;
}

const TypeDescription* get_type_description(short*)
{
  static TypeDescription* td = 0;
  static string nm("short");
  static string path("builtin");
  if(!td){
    td = scinew TypeDescription(nm, path);
  }
  return td;
}

const TypeDescription* get_type_description(int*)
{
  static TypeDescription* td = 0;
  static string nm("int");
  static string path("builtin");
  if(!td){
    td = scinew TypeDescription(nm, path);
  }
  return td;
}

const TypeDescription* get_type_description(unsigned char*)
{
  static TypeDescription* td = 0;
  static string nm("unsigned char");
  static string path("builtin");
  if(!td){
    td = scinew TypeDescription(nm, path);
  }
  return td;
}
const TypeDescription* get_type_description(Vector*)
{
  static TypeDescription* td = 0;
  static string nm("Vector");
  if(!td){
    td = scinew TypeDescription(nm, Vector::get_h_file_path());
  }
  return td;
}
const TypeDescription* get_type_description(Tensor*)
{
  return 0;
}
const TypeDescription* get_type_description(Point*)
{
  return 0;
}
const TypeDescription* get_type_description(Transform*)
{
  return 0;
}

const TypeDescription* get_type_description(string*)
{
  static TypeDescription* td = 0;
  static string nm("string");
  static string path("std::string");
  if(!td){
    td = scinew TypeDescription(nm, path);
  }
  return td;
}

} // end namespace SCIRun
