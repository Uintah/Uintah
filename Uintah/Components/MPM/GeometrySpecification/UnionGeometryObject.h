#ifndef __UNION_GEOMETRY_OBJECT_H__
#define __UNION_GEOMETRY_OBJECT_H__      


#include "GeometryObject.h"
#include <vector>
#include <Uintah/Grid/Box.h>
#include <SCICore/Geometry/Point.h>

using Uintah::Grid::Box;
using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {

/**************************************
	
CLASS
   UnionGeometryObject
	
   Short description...
	
GENERAL INFORMATION
	
   UnionGeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   UnionGeometryObject
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

class UnionGeometryObject : public GeometryObject {

 public:
  //////////
  // Insert Documentation Here:
  UnionGeometryObject(ProblemSpecP &);

  //////////
  // Insert Documentation Here:
  virtual ~UnionGeometryObject();

  //////////
  // Insert Documentation Here:
  virtual bool inside(const Point &p) const;

  //////////
  // Insert Documentation Here:
  virtual Box getBoundingBox() const;

 private:
  std::vector<GeometryObject* > child;

};

} //end namespace Components
} //end namespace Uintah

#endif // __UNION_GEOMETRY_OBJECT_H__
