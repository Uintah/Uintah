#ifndef __DIFFERENCE_GEOMETRY_OBJECT_H__
#define __DIFFERENCE_GEOMETRY_OBJECT_H__      


#include "GeometryObject.h"
#include <Uintah/Grid/Box.h>
#include <SCICore/Geometry/Point.h>


using Uintah::Grid::Box;
using SCICore::Geometry::Point;

namespace Uintah {
namespace Components {

/**************************************
	
CLASS
   DifferenceGeometryObject
	
   Short description...
	
GENERAL INFORMATION
	
   DifferenceGeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   DifferenceGeometryObject
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

class DifferenceGeometryObject : public GeometryObject {

 public:
  //////////
  // Insert Documentation Here:
  DifferenceGeometryObject(ProblemSpecP &);

  //////////
  // Insert Documentation Here:
  virtual ~DifferenceGeometryObject();

  //////////
  // Insert Documentation Here:
  virtual bool inside(const Point &p) const;

  //////////
  // Insert Documentation Here:
  virtual Box getBoundingBox() const;

 private:
  GeometryObject* left;
  GeometryObject* right;


};

} // end namespace Components
} // end namespace Uintah

#endif // __DIFFERENCE_GEOMETRY_OBJECT_H__
