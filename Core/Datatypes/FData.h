// FData.h - the base field data class.
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   January 2001
//
//  Copyright (C) 2001 SCI Institute
//
//  General storage class for Fields.
//

#ifndef SCI_project_FData_h
#define SCI_project_FData_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/TypeName.h>

namespace SCIRun {

class FData : public Datatype 
{
public:

  // GROUP:  Constructors/Destructor
  //////////
  //
  FData();
  virtual ~FData();
  
  // GROUP: Class interface functions
  //////////
  // 
  
  // GROUP: Support of persistent representation
  //////////
  //
  void    io(Piostream&);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) = 0;

protected:
};

typedef LockingHandle<FData> FDataHandle;

}  // end namespace SCIRun

#endif
