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
	
   Creates a collection of geometry objects from the xml input 
   file description. 
	
GENERAL INFORMATION
	
   UnionGeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   UnionGeometryObject BoundingBox inside
	
DESCRIPTION
   Creates a union of different geometry objects from the xml input 
   file description.
   Requires multiple inputs: specify multiple geometry objects.  
   There are methods for checking if a point is inside the union of 
   objects and also for determining the bounding box for the collection.
   The input form looks like this:
       <union>
         <box>
	   <min>[0.,0.,0.]</min>
	   <max>[1.,1.,1.]</max>
	 </box>
	 <sphere>
	   <origin>[.5,.5,.5]</origin>
	   <radius>1.5</radius>
	 </sphere>
       </union>
	
	
WARNING
	
****************************************/

class UnionGeometryObject : public GeometryObject {

 public:
  //////////
  // Constructor that takes a ProblemSpecP argument.   It reads the xml 
  // input specification and builds the intersection of geometry objects.
  UnionGeometryObject(ProblemSpecP &);

  //////////
  // Destructor
  virtual ~UnionGeometryObject();

  //////////
  // Determines whether a point is inside the intersection object.
  virtual bool inside(const Point &p) const;

  //////////
  // Returns the bounding box surrounding the union object.
  virtual Box getBoundingBox() const;

 private:
  std::vector<GeometryObject* > child;

};

} //end namespace Components
} //end namespace Uintah

#endif // __UNION_GEOMETRY_OBJECT_H__

// $Log$
// Revision 1.7  2000/04/22 18:19:11  jas
// Filled in comments.
//
// Revision 1.6  2000/04/22 16:55:12  jas
// Added logging of changes.
//
