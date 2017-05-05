/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
 */

#include <string>

#include <Core/Datatypes/TypeName.h>

namespace Uintah{

using std::string;

class Vector;
class IntVector;
class Point;
class NrrdData;
class Matrix;


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

template<> const string find_type_name(IntVector*)
{
  static const string name = "IntVector";
  return name;
}

template<> const string find_type_name(Point*)
{
  static const string name = "Point";
  return name;
}


template<> const string find_type_name(string*)
{
  static const string name = "string";
  return name;
}



template<> const string find_type_name(LockingHandle<Matrix> *)
{
  static const string name = string("LockingHandle") + FTNS + string("Matrix") + FTNE;
  return name;
}

template<> const string find_type_name(LockingHandle<NrrdData> *)
{
  static const string name = string("LockingHandle") + FTNS + string("NrrdData") + FTNE;
  return name;
}

template<> const string find_type_name(LockingHandle<Field> *)
{
  static const string name = string("LockingHandle") + FTNS + string("Field") + FTNE;
  return name;
}

} // namespace Uintah
