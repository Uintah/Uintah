/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
  static const string name = "float";
  return name;
}

template<> const string find_type_name(double*)
{
  static const string name = "double";
  return name;
}

template<> const string find_type_name(long double*)
{
  static const string name = "long_double";
  return name;
}

//////////
// Integer types
template<> const string find_type_name(short*)
{
  static const string name = "short";
  return name;
}

template<> const string find_type_name(unsigned short*)
{
  static const string name = "unsigned_short";
  return name;
}

template<> const string find_type_name(int*)
{
  static const string name = "int";
  return name;
}

template<> const string find_type_name(unsigned int*)
{
  static const string name = "unsigned_int";
  return name;
}

template<> const string find_type_name(long*)
{
  static const string name = "long";
  return name;
}

template<> const string find_type_name(unsigned long*)
{
  static const string name = "unsigned_long";
  return name;
}

template<> const string find_type_name(long long*)
{
  static const string name = "long_long";
  return name;
}

template<> const string find_type_name(unsigned long long*)
{
  static const string name = "unsigned_long_long";
  return name;
}

//////////
// Character types
template<> const string find_type_name(char*)
{
  static const string name = "char";
  return name;
}

template<> const string find_type_name(unsigned char*)
{
  static const string name = "unsigned_char";
  return name;
}

//////////
// Boolean type
template<> const string find_type_name(bool*)
{
  static const string name = "bool";
  return name;
}

//////////
// Template function specializations for some primitives
template<> const string find_type_name(Vector*)
{
  static const string name = "Vector";
  return name;
}

template<> const string find_type_name(Point*)
{
  static const string name = "Point";
  return name;
}

template<> const string find_type_name(Transform*)
{
  static const string name = "Transform";
  return name;
}

template<> const string find_type_name(string*)
{
  static const string name = "string";
  return name;
}

} // namespace SCIRun
