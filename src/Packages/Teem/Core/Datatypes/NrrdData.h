/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

// NrrdData.h - interface to Gordon's Nrrd class
//
//  Written by:
//   David Weinstein
//   School of Computing
//   University of Utah
//   February 2001
//
//  Copyright (C) 2001 SCI Institute

#if !defined(SCI_Teem_NrrdData_h)
#define SCI_Teem_NrrdData_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Containers/LockingHandle.h>
#include <nrrd.h>

namespace SCITeem {

using namespace SCIRun;

/////////
// Structure to hold NrrdData
class NrrdData : public Datatype {
public:  
  // GROUP: public data
  //////////
  // 
  Nrrd *nrrd;
  string fname;

  NrrdData(bool owned = true);
  NrrdData(const NrrdData&);
  ~NrrdData();

  void set_orig_mesh(MeshHandle mh) { originating_mesh_ = mh; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
private:
  //! did we wrap some existing memory, or was this allocated
  //! for this object to delete.
  bool                 data_owned_;
  //! a handle to the mesh this data originally belonged with. 
  //! has a rep == 0 if there was no such mesh.
  MeshHandle           originating_mesh_; 
};

typedef LockingHandle<NrrdData> NrrdDataHandle;
} // end namespace SCITeem

#endif // SCI_Teem_NrrdData_h
