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


// NrrdData.cc - Interface to Gordon's Nrrd package
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#include <Core/Datatypes/NrrdData.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;

namespace SCIRun {

NrrdData::NrrdData() : 
  nrrd_(nrrdNew()),
  write_nrrd_(true),
  embed_object_(false),
  data_owner_(0)
{
}


NrrdData::NrrdData(Nrrd *n) :
  nrrd_(n),
  write_nrrd_(true),
  embed_object_(false),
  data_owner_(0)
{
}


NrrdData::NrrdData(LockingHandle<Datatype> data_owner) : 
  nrrd_(nrrdNew()),
  write_nrrd_(true),
  embed_object_(false),
  data_owner_(data_owner)
{
}


NrrdData::NrrdData(const NrrdData &copy) :
  nrrd_(nrrdNew()),
  data_owner_(0),
  nrrd_fname_(copy.nrrd_fname_)
{
  nrrdCopy(nrrd_, copy.nrrd_);
}


NrrdData::~NrrdData()
{
  if(!data_owner_.get_rep())
  {
    nrrdNuke(nrrd_);
  }
  else
  {
    nrrdNix(nrrd_);
    data_owner_ = 0;
  }
}


NrrdData* 
NrrdData::clone() 
{
  return new NrrdData(*this);
}


// This would be much easier to check with a regular expression lib
// A valid label has the following format:
// type = one of the valid types (Scalar, Vector, Tensor)
// elem = [A-Za-z0-9\-]+:type
// (elem,?)+

bool 
NrrdData::in_name_set(const string &s) const
{
  for (string::size_type i = 0; i < s.size(); i++)
  {
    if (!(isalnum(s[i]) || s[i] == '-' || s[i] == '_'))
    {
      return false;
    }
  }
  return true;
}


template <>
unsigned int get_nrrd_type<Tensor>()
{
  return nrrdTypeFloat;
}


template <>
unsigned int get_nrrd_type<char>()
{
  return nrrdTypeChar;
}


template <>
unsigned int get_nrrd_type<unsigned char>()
{
  return nrrdTypeUChar;
}


template <>
unsigned int get_nrrd_type<short>()
{
  return nrrdTypeShort;
}


template <>
unsigned int get_nrrd_type<unsigned short>()
{
  return nrrdTypeUShort;
}


template <>
unsigned int get_nrrd_type<int>()
{
  return nrrdTypeInt;
}


template <>
unsigned int get_nrrd_type<unsigned int>()
{
  return nrrdTypeUInt;
}


template <>
unsigned int get_nrrd_type<long long>()
{
  return nrrdTypeLLong;
}


template <>
unsigned int get_nrrd_type<unsigned long long>()
{
  return nrrdTypeULLong;
}


template <>
unsigned int get_nrrd_type<float>()
{
  return nrrdTypeFloat;
}


void get_nrrd_compile_type( const unsigned int type,
			    string & typeStr,
			    string & typeName )
{
  switch (type) {
  case nrrdTypeChar :  
    typeStr = string("char");
    typeName = string("char");
    break;
  case nrrdTypeUChar : 
    typeStr = string("unsigned char");
    typeName = string("unsigned_char");
    break;
  case nrrdTypeShort : 
    typeStr = string("short");
    typeName = string("short");
    break;
  case nrrdTypeUShort :
    typeStr = string("unsigned short");
    typeName = string("unsigned_short");
    break;
  case nrrdTypeInt : 
    typeStr = string("int");
    typeName = string("int");
    break;
  case nrrdTypeUInt :  
    typeStr = string("unsigned int");
    typeName = string("unsigned_int");
    break;
  case nrrdTypeLLong : 
    typeStr = string("long long");
    typeName = string("long_long");
    break;
  case nrrdTypeULLong :
    typeStr = string("unsigned long long");
    typeName = string("unsigned_long_long");
    break;
  case nrrdTypeFloat :
    typeStr = string("float");
    typeName = string("float");
    break;
  case nrrdTypeDouble :
    typeStr = string("double");
    typeName = string("double");
    break;
  default:
    typeStr = string("float");
    typeName = string("float");
  }
}


unsigned int
string_to_nrrd_type(const string &str)
{
  if (str == "nrrdTypeChar")
    return nrrdTypeChar;
  else if (str == "nrrdTypeUChar")
    return nrrdTypeUChar;
  else if (str == "nrrdTypeShort")
    return nrrdTypeShort;
  else if (str == "nrrdTypeUShort")
    return nrrdTypeUShort;
  else if (str == "nrrdTypeInt")
    return nrrdTypeInt;
  else if (str == "nrrdTypeUInt")
    return nrrdTypeUInt;
  else if (str == "nrrdTypeLLong")
    return nrrdTypeLLong;
  else if (str == "nrrdTypeULLong")
    return nrrdTypeULLong;
  else if (str == "nrrdTypeFloat")
    return nrrdTypeFloat;
  else if (str == "nrrdTypeDouble")
    return nrrdTypeDouble;
  else
  {
    ASSERTFAIL("Unknown nrrd string type");
    return nrrdTypeFloat;
  }
}

}  // end namespace SCIRun
