//  RegLatticeGeom.h - A lattice with equally spaced axis in 1,
//  2, or 3 dimensions
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute


#ifndef SCI_project_RegLatticeGeom_h
#define SCI_project_RegLatticeGeom_h 1

#include <SCICore/Datatypes/LatticeGeom.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <vector>
#include <string>


namespace SCICore {
namespace Datatypes{

using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using std::vector;
using std::string;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;


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


} // end namespace Datatypes
} // end namespace SCICore
  

#endif
