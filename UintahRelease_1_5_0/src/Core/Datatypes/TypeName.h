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


/*
 *   TypeName.h : template to return name of argument type; 
 *                used in PIO of templatized types
 *                
 *   Created by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *   Copyright (C) 2000 SCI Institute
 *   
 */

#include <string>
#include <vector>
#include <map>
#include <Core/Datatypes/FieldIndex.h>
#include <Core/Containers/LockingHandle.h>

#include <Core/Datatypes/share.h>

#ifndef TYPENAME_H
#define TYPENAME_H

namespace SCIRun {

using std::string;
using std::vector;
using std::pair;

static const char FTNS = '<';
static const char FTNM = ',';
static const char FTNE = '>';


//////////
// Function to return name of type of its argument
template <class T> const string find_type_name(T*)
{
  return T::type_name(-1);
}


template<class T, class S> const string find_type_name( pair<T,S> *);

class Vector;
class IntVector;
class Point;
class Transform;
class Matrix;
class NrrdData;
class Field;

template<> SCISHARE const string find_type_name(float*);
template<> SCISHARE const string find_type_name(double*);
template<> SCISHARE const string find_type_name(long double*);
template<> SCISHARE const string find_type_name(short*);
template<> SCISHARE const string find_type_name(unsigned short*);
template<> SCISHARE const string find_type_name(int*);
template<> SCISHARE const string find_type_name(unsigned int*);
template<> SCISHARE const string find_type_name(long*);
template<> SCISHARE const string find_type_name(unsigned long*);
template<> SCISHARE const string find_type_name(long long*);
template<> SCISHARE const string find_type_name(unsigned long long*);
template<> SCISHARE const string find_type_name(char*);
template<> SCISHARE const string find_type_name(unsigned char*);
template<> SCISHARE const string find_type_name(bool*);
template<> SCISHARE const string find_type_name(Vector*);
template<> SCISHARE const string find_type_name(IntVector*);
template<> SCISHARE const string find_type_name(Point*);
template<> SCISHARE const string find_type_name(Transform*);
template<> SCISHARE const string find_type_name(string*);

template<> SCISHARE const string find_type_name(LockingHandle<Matrix> *);
template<> SCISHARE const string find_type_name(LockingHandle<NrrdData> *);
template<> SCISHARE const string find_type_name(LockingHandle<Field> *);

//////////
// Function overloading for templates 
template<class T> class Array1;
template<class T> class Array2;

template <class T> const string find_type_name(Array1<T>*)
{
  static const string name = string("Array1") + FTNS + find_type_name((T*)0) + FTNE;
  return name;
}

template <class T> const string find_type_name(Array2<T>*)
{
  static const string name = string("Array2") + FTNS + find_type_name((T*)0) + FTNE;
  return name;
}

template <class T> const string find_type_name(vector<T>*)
{
  static const string name = string("vector") + FTNS + find_type_name((T*)0) + FTNE;
  return name;
}

template<class T, class S> const string find_type_name( pair<T,S> *)
{
  static const string name = string("pair") + FTNS + find_type_name((T*)0) + FTNM + find_type_name((S*)0) + FTNE;
  return name;
}

} // namespace SCIRun

#endif
