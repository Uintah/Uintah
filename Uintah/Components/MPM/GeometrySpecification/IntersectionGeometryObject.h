#ifndef __INTERSECTION_GEOMETRY_OBJECT_H__
#define __INTERSECTION_GEOMETRY_OBJECT_H__      


#include "GeometryObject.h"
#include <Uintah/Grid/Box.h>
#include <vector>
#include <SCICore/Geometry/Point.h>


using Uintah::Grid::Box;
using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {

/**************************************
	
CLASS
   IntersectionGeometryObject
	
   Short description...
	
GENERAL INFORMATION
	
   IntersectionGeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   IntersectionGeometryObject
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

class IntersectionGeometryObject : public GeometryObject {

 public:
  //////////
  // Insert Documentation Here:
  IntersectionGeometryObject(ProblemSpecP &);

  //////////
  // Insert Documentation Here:
  virtual ~IntersectionGeometryObject();

  //////////
  // Insert Documentation Here:  
  virtual bool inside(const Point &p) const;

  //////////
  // Insert Documentation Here:
  virtual Box getBoundingBox() const;

 private:
  std::vector<GeometryObject* > child;

};

} // end namespace Components
} // end namespace Uintah

#endif // __INTERSECTION_GEOMETRY_OBJECT_H__

// $Log$
// Revision 1.4  2000/04/22 16:55:12  jas
// Added logging of changes.
//
