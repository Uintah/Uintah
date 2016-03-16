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
//    File   : TypeDescription.cc
//    Author : Martin Cole
//    Date   : Mon May 14 10:20:21 2001

#include <Core/Util/TypeDescription.h>
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

  static map<string, const SCIRun::STypeDescription*>* types = 0;
  static vector<const SCIRun::STypeDescription*>* typelist=0;
  static Uintah::Mutex typelist_lock("TypeDescription::typelist lock");
static bool killed=false;

KillMap::~KillMap()
{
  if(!types){
    ASSERT(!killed);
    ASSERT(!typelist);
    return;
  }
  killed=true;
  vector<const SCIRun::STypeDescription*>::iterator iter = typelist->begin();
  for(;iter != typelist->end();iter++)
    delete *iter;
  delete types;
  delete typelist;
}

KillMap killit;


void
SCIRun::STypeDescription::register_type()
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
      types = scinew map<string, const SCIRun::STypeDescription*>;
      typelist = scinew vector<const STypeDescription*>;
    }
  }

  map<string, const STypeDescription*>::iterator iter = types->find(get_name());
  if (iter == types->end())
  {
    (*types)[get_name()] = this;
  }
  typelist->push_back(this);
  
  typelist_lock.unlock();
}


STypeDescription::STypeDescription(const string &name, category_e c) : 
  subtype_(0), 
  name_(name),
  category_(c)
{
  register_type();
}

STypeDescription::STypeDescription(const string &name, td_vec* sub, category_e c) : 
  subtype_(sub),
  name_(name),
  category_(c)
{
  register_type();
}

STypeDescription::~STypeDescription()
{
  if (subtype_) delete subtype_;
}

string 
STypeDescription::get_name( const string & type_sep_start /* = "<"  */, 
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
STypeDescription::get_similar_name(const string &substitute,
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
STypeDescription::get_filename() const
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


const STypeDescription* 
STypeDescription::lookup_type(const std::string& t)
{
  if(!types) {
    typelist_lock.lock();
    if (!types) {
      types=scinew map<string, const STypeDescription*>;
      typelist=new vector<const STypeDescription*>;
    }
    typelist_lock.unlock();
  }
  
  map<string, const STypeDescription*>::iterator iter = types->find(t);
   if(iter == types->end())
      return 0;
   return iter->second;
}



STypeDescription::Register::Register(const STypeDescription* /* td*/)
{
  // Registration happens in CTOR
}

STypeDescription::Register::~Register()
{
}

const STypeDescription* get_type_description(double*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("double");
  }
  return td;
}

const STypeDescription* get_type_description(long*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("long");
  }
  return td;
}

const STypeDescription* get_type_description(float*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("float");
  }
  return td;
}

const STypeDescription* get_type_description(short*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("short");
  }
  return td;
}

const STypeDescription* get_type_description(unsigned short*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("unsigned short");
  }
  return td;
}

const STypeDescription* get_type_description(int*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("int");
  }
  return td;
}

const STypeDescription* get_type_description(unsigned int*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("unsigned int");
  }
  return td;
}

const STypeDescription* get_type_description(char*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("char");
  }
  return td;
}

const STypeDescription* get_type_description(unsigned char*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("unsigned char");
  }
  return td;
}

const STypeDescription* get_type_description(string*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("string");
  }
  return td;
}

const STypeDescription* get_type_description(unsigned long*)
{
  static STypeDescription* td = 0;
  if(!td){
    td = scinew STypeDescription("unsigned long");
  }
  return td;
}

} // end namespace Uintah
