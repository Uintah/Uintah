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
	
   Creates the intersection of geometry objects from the xml input 
   file description. 

	
GENERAL INFORMATION
	
   IntersectionGeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   IntersectionGeometryObject
	
DESCRIPTION
   Creates a intersection of different geometry objects from the xml input 
   file description.
   Requires multiple inputs: specify multiple geometry objects.  
   There are methods for checking if a point is inside the intersection of 
   objects and also for determining the bounding box for the collection.
   The input form looks like this:
       <intersection>
         <box>
	   <min>[0.,0.,0.]</min>
	   <max>[1.,1.,1.]</max>
	 </box>
	 <sphere>
	   <origin>[.5,.5,.5]</origin>
	   <radius>1.5</radius>
	 </sphere>
       </intersection>
	
WARNING
	
****************************************/

class IntersectionGeometryObject : public GeometryObject {

 public:
  //////////
  // Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds the intersection of geometry objects.
  IntersectionGeometryObject(ProblemSpecP &);

  //////////
  // Destructor
  virtual ~IntersectionGeometryObject();

  //////////
  // Determines whether a point is inside the intersection object.  
  virtual bool inside(const Point &p) const;

  //////////
  // Returns the bounding box surrounding the intersection object.
  virtual Box getBoundingBox() const;

 private:
  std::vector<GeometryObject* > child;

};

} // end namespace Components
} // end namespace Uintah

#endif // __INTERSECTION_GEOMETRY_OBJECT_H__

// $Log$
// Revision 1.5  2000/04/22 18:19:11  jas
// Filled in comments.
//
// Revision 1.4  2000/04/22 16:55:12  jas
// Added logging of changes.
//
