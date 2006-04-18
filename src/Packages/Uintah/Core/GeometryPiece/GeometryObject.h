
#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <sgi_stl_warnings_off.h>
#include   <list>
#include   <string>
#include   <map>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

class GeometryPiece;

using namespace SCIRun;
using std::string;
using std::list;
using std::map;

/**************************************
	
CLASS
   GeometryObject
	
   Short description...
	
GENERAL INFORMATION
	
   GeometryObject.h
	
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
	
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
 
	
KEYWORDS
   GeometryObject
	
DESCRIPTION
   Long description...
	
WARNING
	
****************************************/

class GeometryObject {
	 
public:
  //////////
  // Insert Documentation Here:
  GeometryObject(GeometryPieceP piece, ProblemSpecP&,list<string>& data);

  //////////
  // Insert Documentation Here:
  ~GeometryObject() {}

  void outputProblemSpec(ProblemSpecP& ps);

  //////////
  // Insert Documentation Here:
  IntVector getNumParticlesPerCell() const { return d_resolution; }

  //////////
  // Insert Documentation Here:
  GeometryPieceP getPiece() const {
    return d_piece;
  }

  Vector getInitialVelocity() const {
    return d_initialVel;
  }

  double getInitialData(const string& data_string) {
    return d_data[data_string];
  }

private:
  GeometryPieceP     d_piece;
  IntVector          d_resolution;
  Vector             d_initialVel;
  map<string,double> d_data;

};

} // End namespace Uintah
      
#endif // __GEOMETRY_OBJECT_H__

