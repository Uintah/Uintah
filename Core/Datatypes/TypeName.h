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

#ifndef TYPENAME_H
#define TYPENAME_H

namespace SCIRun{

using std::string;

//////////
// Function to return name of type of its argument

template <class T> string findTypeName(T*){
  return T::typeName();
}

class Vector;
class Point;

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

} // namespace SCIRun

#endif
