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
	
   Creates the difference between two geometry objects from the xml input 
   file description. 


GENERAL INFORMATION
	
   DifferenceGeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   DifferenceGeometryObject BoundingBox inside
	
DESCRIPTION
   Creates the difference between two  geometry objects from the xml input 
   file description.
   Requires tow inputs: specify two geometry objects. The order is important.
   There are methods for checking if a point is inside the difference of 
   objects and also for determining the bounding box for the collection.
   The input form looks like this:
       <difference>
         <box>
	   <min>[0.,0.,0.]</min>
	   <max>[1.,1.,1.]</max>
	 </box>
	 <sphere>
	   <origin>[.5,.5,.5]</origin>
	   <radius>1.5</radius>
	 </sphere>
       </difference>

	
WARNING
	
****************************************/

class DifferenceGeometryObject : public GeometryObject {

 public:
  //////////
  //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds the union of geometry objects.
  DifferenceGeometryObject(ProblemSpecP &);

  //////////
  // Destructor
  virtual ~DifferenceGeometryObject();

  //////////
  // Determines whether a point is inside the union object.
  virtual bool inside(const Point &p) const;

  //////////
  // Returns the bounding box surrounding the union object.
  virtual Box getBoundingBox() const;

 private:
  GeometryObject* left;
  GeometryObject* right;


};

} // end namespace Components
} // end namespace Uintah

#endif // __DIFFERENCE_GEOMETRY_OBJECT_H__

// $Log$
// Revision 1.5  2000/04/22 18:19:11  jas
// Filled in comments.
//
// Revision 1.4  2000/04/22 16:55:12  jas
// Added logging of changes.
//
