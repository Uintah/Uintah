//  GenSField.cc - A general scalar field, comprised of one attribute and one geometry
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/GenSField.h>

namespace SCICore{
namespace Datatypes{

template <class T> PersistentTypeID GenSField<T>::type_id("GenSField", "Datatype", 0);

template <class T> GenSField<T>::GenSField():
  SField(){
}

template <class T> GenSField<T>::~GenSField(){
}

template <class T> GenSField<T>::GenSField(Geom* igeom, Attrib* iattrib):
  SField(){
  geom = 0;
  attrib = 0;
  geom = igeom;
  attrib = iattrib;
}

template <class T> GenSField<T>::GenSField(const GenSField&){
}

template <class T> void GenSField<T>::resize(int a, int b, int c){
}

template <class T> T& GenSField<T>::grid(const Point&){
  return *(new T);
}

template <class T> T& GenSField<T>::operator[](int){
  return *(new T);
}

template <class T> void GenSField<T>::set_bounds(const Point&, const Point&){
}

template <class T> FieldInterface* GenSField<T>::query_interface(const string& istring){
  if(istring == "sinterpolate"){
    return dynamic_cast<SInterpolate<T>*>(this);
  }
  else{
    // nothing matched, call parent class
    return Field::query_interface(istring);
  }
}

template <class T> int GenSField<T>::sinterpolate(const Point& ipoint, T& outval){

  return 0;
}

template <class T> void GenSField<T>::io(Piostream&){
}



} // end Datatypes
} // end SCICore

