//  VField.h - Vector Field
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute

#ifndef SCI_project_VField_h
#define SCI_project_VField_h 1

#include <Core/Datatypes/Field.h>
#include <Core/Containers/LockingHandle.h>


namespace SCIRun {


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


} // End namespace SCIRun

#endif
