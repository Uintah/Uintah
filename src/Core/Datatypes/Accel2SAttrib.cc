//  FlatSAttrib.cc - scalar attribute stored as a flat array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Accel2SAttrib.h>

namespace SCICore {
namespace Datatypes {

template <class T> PersistentTypeID Accel2SAttrib<T>::type_id("Accel2SAttrib", "Datatype", 0);

template <class T> Accel2SAttrib<T>::Accel2SAttrib(){
}

template <class T> Accel2SAttrib<T>::~Accel2SAttrib(){
}

template <class T> void Accel2SAttrib<T>::io(Piostream&){
}


} // end Datatypes
} // end SCICore
