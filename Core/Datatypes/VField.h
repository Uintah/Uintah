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

class VField;
typedef LockingHandle<VField> VFieldHandle;

class VField : public Field {
public:
  VField();

  virtual ~VField();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:
};


} // end SCICore
} // end Datatypes

#endif
