/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef __GEOMETRY_OBJECT_H__
#define __GEOMETRY_OBJECT_H__

#include <Core/Geometry/IntVector.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include   <list>
#include   <string>
#include   <map>

#include <Core/GeometryPiece/uintahshare.h>
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

class UINTAHSHARE GeometryObject {
	 
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

