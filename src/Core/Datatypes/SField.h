//  SField.h - Scalar Field
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_SField_h
#define SCI_project_SField_h 1


#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/FlatAttrib.h>

namespace SCIRun {


class SField;
typedef LockingHandle<SField> SFieldHandle;

class SCICORESHARE SField:public Field{
public:
   //////////
  // Constructor
  SField();

  //////////
  // Destructor
  virtual ~SField();

  //////////
  // Used by SFieldHandle to obtain a SField*
  //inline SField* get_base() {return this;};

  virtual bool get_minmax(double &imin, double &imax) = 0;

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
private:
  bool has_minmax;
};

} // End namespace SCIRun

#endif
