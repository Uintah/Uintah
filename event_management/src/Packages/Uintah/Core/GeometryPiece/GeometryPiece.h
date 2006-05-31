#ifndef __GEOMETRY_PIECE_H__
#define __GEOMETRY_PIECE_H__

#include <Packages/Uintah/Core/Util/RefCounted.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <sgi_stl_warnings_off.h>
#include   <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace Uintah {

using namespace SCIRun;

class Box;
class GeometryPiece;

template<class T> class Handle;
typedef Handle<GeometryPiece> GeometryPieceP;

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
      
class GeometryPiece : public RefCounted {
	 
public:
  //////////
  // Insert Documentation Here:
  GeometryPiece();
	 
  //////////
  // Insert Documentation Here:
  virtual ~GeometryPiece();

  /// Clone a geometry piece
  virtual GeometryPieceP clone() const = 0;

  void outputProblemSpec( ProblemSpecP & ps ) const;

  //////////
  // Insert Documentation Here:
  virtual Box getBoundingBox() const = 0;
	 
    //////////
    // Insert Documentation Here:
  virtual bool inside(const Point &p) const = 0;	 

  std::string getName() const {
    return name_;
  }

  // Returns the type as a string (eg: sphere, box, etc).
  // The string must/will match the string checks found in GeometryPieceFactory.cc::create()
  virtual std::string getType() const = 0;

  void setName(const std::string& name) {
    nameSet_ = true;
    name_    = name;
  }

  // Call at the beginning of outputing (ProblemSpec) so that this
  // object will output the full spec the first time, and only a
  // reference subsequently.
  void resetOutput() const { firstOutput_ = true; }

protected:

  virtual void outputHelper( ProblemSpecP & ps ) const = 0;

  bool        nameSet_; // defaults to false
  std::string name_;

  // Used for outputing the problem spec... on the 1st output, the
  // entire object is output, on the 2nd, only a reference is output.
  // Must be 'mutable' as most of these objects are (mostly) 'const'
  mutable bool firstOutput_;

};

} // End namespace Uintah

#endif // __GEOMETRY_PIECE_H__
