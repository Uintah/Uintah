//  VField.h - Vector Field
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_VField_h
#define SCI_project_VField_h 1

#include <SCICore/Datatypes/Field.h>
#include <SCICore/Containers/LockingHandle.h>


namespace SCICore{
namespace Datatypes{

using SCICore::Containers::LockingHandle;

class Object;

template <class T> class VField;
typedef LockingHandle<VField<Object> > VFieldHandle;

template <class T> class VField:public Field{
public:
  VField();
  VField(const Geom&, const Attrib&);
  VField(const VField&);
  ~VField();

  virtual void Interpolate(T&, const Point&);
  virtual bool gradient(Vector&, const Point&);
  virtual T& grid(int, int, int);
  virtual T& operator[](int);
  
  
private:
};


} // end SCICore
} // end Datatypes

#endif
