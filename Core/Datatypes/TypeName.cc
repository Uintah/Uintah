/*
 *   TypeName.h :  specializations of template<class T> 
 *                 find_type_name() function for build-in 
 *                 and simple types not deriving from Core::Datatype
 *                 
 *   Created by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *   Copyright (C) 2000 SCI Institute
 */

#include <string>

#include <Core/Datatypes/TypeName.h>

namespace SCIRun{

using std::string;

class Vector;
class Point;

//////////
// Template function specializations for built-in types

//////////
// Floating-point types
template<> const string find_type_name(float*)
{
  return "float";
}

template<> const string find_type_name(double*)
{
  return "double";
}

template<> const string find_type_name(long double*)
{
  return "long double";
}

//////////
// Integer types
template<> const string find_type_name(short*)
{
  return "short";
}

template<> const string find_type_name(unsigned short*)
{
  return "unsigned short";
}

template<> const string find_type_name(int*)
{
  return "int";
}

template<> const string find_type_name(unsigned int*)
{
  return "unsigned int";
}

template<> const string find_type_name(long*)
{
  return "long";
}

template<> const string find_type_name(unsigned long*)
{
  return "unsigned long";
}

template<> const string find_type_name(long long*)
{
  return "long long";
}

template<> const string find_type_name(unsigned long long*)
{
  return "unsigned long long";
}

//////////
// Character types
template<> const string find_type_name(char*)
{
  return "char";
}

template<> const string find_type_name(unsigned char*)
{
  return "unsigned char";
}

//////////
// Boolean type
template<> const string find_type_name(bool*)
{
  return "bool";
}

//////////
// Template function specializations for some primitives
template<> const string find_type_name(Vector*)
{
  return "Vector";
}

template<> const string find_type_name(Point*)
{
  return "Point";
}

template<> const string find_type_name(Transform*)
{
  return "Transform";
}

template<> const string find_type_name(string*)
{
  return "string";
}

} // namespace SCIRun
