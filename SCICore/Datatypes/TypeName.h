/*
 *   TypeName.h : template to return name of argument type 
 *               
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

#ifndef TYPENAME_H
#define TYPENAME_H

namespace SCICore{
namespace Datatypes{

//////////
// Function to return name of type of its argument
template <class T> string findTypeName(T){
  return T::typeName();
}

} // namespace Datatypes
} // namespace SCICore

#endif
