//  FlatSAttrib.cc - scalar attribute stored as a flat array
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/FlatSAttrib.h>

namespace SCICore{
namespace Datatypes{

template <class T> PersistentTypeID FlatSAttrib<T>::type_id("FlatSAttrib", "Datatype", 0);

template <class T> FlatSAttrib<T>::FlatSAttrib(){
}

template <class T> FlatSAttrib<T>::~FlatSAttrib(){
}

template <class T> void FlatSAttrib<T>::io(Piostream&){
}


} // end Datatypes
} // end SCICore
