// NrrdData.h - interface to Gordon's Nrrd class
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#ifndef SCI_Nrrd_NrrdData_h
#define SCI_Nrrd_NrrdData_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/String.h>
#include <nrrd.h>

namespace SCINrrd {

using namespace SCIRun;

class NrrdData;
typedef LockingHandle<NrrdData> NrrdDataHandle;

/////////
// Structure to hold NrrdData
class NrrdData : public Datatype {
public:  
  // GROUP: public data
  //////////
  // 
  Nrrd *nrrd;
  clString fname;

  NrrdData();
  NrrdData(const NrrdData&);
  ~NrrdData();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};
} // end namespace SCINrrd

#endif
