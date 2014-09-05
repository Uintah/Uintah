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

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <map>
#include <sgi_stl_warnings_on.h>
#include <Core/Datatypes/FieldIndex.h>

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
class Point;
class Transform;

template<> const string find_type_name(float*);
template<> const string find_type_name(double*);
template<> const string find_type_name(long double*);
template<> const string find_type_name(short*);
template<> const string find_type_name(unsigned short*);
template<> const string find_type_name(int*);
template<> const string find_type_name(unsigned int*);
template<> const string find_type_name(long*);
template<> const string find_type_name(unsigned long*);
template<> const string find_type_name(long long*);
template<> const string find_type_name(unsigned long long*);
template<> const string find_type_name(char*);
template<> const string find_type_name(unsigned char*);
template<> const string find_type_name(bool*);
template<> const string find_type_name(Vector*);
template<> const string find_type_name(Point*);
template<> const string find_type_name(Transform*);
template<> const string find_type_name(string*);

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
