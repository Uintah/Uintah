//  FlatSAttrib.cc - scalar attribute stored as a flat array
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   August 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Accel3SAttrib.h>

namespace SCICore {
namespace Datatypes {

template <class T> PersistentTypeID Accel3SAttrib<T>::type_id("Accel3SAttrib", "Datatype", 0);

template <class T> Accel3SAttrib<T>::Accel3SAttrib(){
}

template <class T> Accel3SAttrib<T>::~Accel3SAttrib(){
}

template <class T> void Accel3SAttrib<T>::io(Piostream&){
}

} // end Datatypes
} // end SCICore
