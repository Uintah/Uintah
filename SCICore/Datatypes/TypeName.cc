/*
 *   TypeName.h :  specializations of template<class T> 
 *                 findTypeName() function for build-in types
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
using namespace std;

#include <SCICore/Datatypes/TypeName.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore{
namespace Datatypes{

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

//////////
// Template function specializations for built-in types

//////////
// Floating-point types
template<> string findTypeName(float){
  return string("float");
}

template<> string findTypeName(double){
  return string("double");
}

template<> string findTypeName(long double){
  return string("long double");
}

//////////
// Integer types
template<> string findTypeName(short){
  return string("short");
}

template<> string findTypeName(unsigned short){
  return string("unsigned short");
}

template<> string findTypeName(int){
  return string("int");
}

template<> string findTypeName(unsigned int){
  return string("unsigned int");
}

template<> string findTypeName(long){
  return string("long");
}

template<> string findTypeName(unsigned long){
  return string("unsigned long");
}

template<> string findTypeName(long long){
  return string("long long");
}

template<> string findTypeName(unsigned long long){
  return string("unsigned long long");
}

//////////
// Character types
template<> string findTypeName(char){
  return string("char");
}

template<> string findTypeName(unsigned char){
  return string("unsigned char");
}

//////////
// Boolean type
template<> string findTypeName(bool){
  return string("bool");
}

//////////
// Template function specializations for some primitives
template<> string findTypeName(Vector){
  return string("Vector");
}

template<> string findTypeName(Point){
  return string("Point");
}

} // namespace Datatypes
} // namespace SCICore
