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

#ifndef TYPENAME_H
#define TYPENAME_H

namespace SCIRun{

using std::string;
using std::vector;
using std::map;

//////////
// Function to return name of type of its argument
template <class T> string findTypeName(T*){
  return T::typeName();
}

class Vector;
class Point;
class Transform;

template<> string findTypeName(float*);
template<> string findTypeName(double*);
template<> string findTypeName(long double*);
template<> string findTypeName(short*);
template<> string findTypeName(unsigned short*);
template<> string findTypeName(int*);
template<> string findTypeName(unsigned int*);
template<> string findTypeName(long*);
template<> string findTypeName(unsigned long*);
template<> string findTypeName(long long*);
template<> string findTypeName(unsigned long long*);
template<> string findTypeName(char*);
template<> string findTypeName(unsigned char*);
template<> string findTypeName(bool*);
template<> string findTypeName(Vector*);
template<> string findTypeName(Point*);
template<> string findTypeName(Transform*);

//////////
// Function overloading for templates 
template<class T> class Array1;
template<class T> class Array2;

template <class T> string findTypeName(Array1<T>*){
  return "Array1<"+findTypeName((T*)0)+">";
}

template <class T> string findTypeName(Array2<T>*){
  return "Array2<"+findTypeName((T*)0)+">";
}

template <class T> string findTypeName(vector<T>*){
  return "vector<"+findTypeName((T*)0)+">";
}

/*
template <class T, class S> string findTypeName(map<T, S>*){
  return "map<"+findTypeName((T*)0)+","+findTypeName((S*)0)+">";
}
*/

} // namespace SCIRun

#endif
