//  RegLatticeGeom.h - A lattice with equally spaced axis in 1,
//  2, or 3 dimensions
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_RegLatticeGeom_h
#define SCI_project_RegLatticeGeom_h 1

#include <Core/Datatypes/LatticeGeom.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <vector>
#include <string>


namespace SCIRun {

using std::vector;
using std::string;


class SCICORESHARE RegLatticeGeom:public LatticeGeom{  
public:

  RegLatticeGeom();
  ~RegLatticeGeom();

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  string get_info();

protected:
  
};


} // End namespace SCIRun
  

#endif
