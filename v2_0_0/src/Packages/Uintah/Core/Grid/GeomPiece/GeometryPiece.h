#ifndef __GEOMETRY_PIECE_H__
#define __GEOMETRY_PIECE_H__

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {
  class Box;
using namespace SCIRun;

/**************************************
	
CLASS
   GeometryPiece
	
   Short description...
	
GENERAL INFORMATION
	
   GeometryPiece.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   GeometryPiece
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

      
      class GeometryPiece {
	 
      public:
	 //////////
	 // Insert Documentation Here:
	 GeometryPiece();
	 
	 //////////
	 // Insert Documentation Here:
	 virtual ~GeometryPiece();
	 
	 //////////
	 // Insert Documentation Here:
	 virtual Box getBoundingBox() const = 0;
	 
	 //////////
	 // Insert Documentation Here:
	 virtual bool inside(const Point &p) const = 0;	 

	 std::string getName() const {
	   return name_;
	 }
	 void setName(const std::string& name) {
	   name_=name;
	 }
      protected:
	 std::string name_;
      };
} // End namespace Uintah

#endif // __GEOMETRY_PIECE_H__
