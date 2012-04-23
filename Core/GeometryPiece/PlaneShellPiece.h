/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef __PLANE_SHELL_PIECE_H__
#define __PLANE_SHELL_PIECE_H__

#include <Core/GeometryPiece/ShellGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <cmath>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/////////////////////////////////////////////////////////////////////////////
/*! 
  \class PlaneShellPiece
	
  \brief Creates a plane shell from the xml input file description.
	
  \author  Biswajit Banerjee \n
   C-SAFE and Department of Mechanical Engineering \n
   University of Utah \n
	
   Creates a plane circle from the xml input file description.
   The input form looks like this:
   \verbatim
     <plane>
       <center>[0.,0.,0.]</center>
       <normal>[0.,0.,1.]</normal>
       <radius>2.0</radius>
       <thickness>0.1</thickness>
       <num_radial>20</num_radial>
     </plane>
   \endverbatim
	
   \warning Needs to be converted into the base class for classes such as
   TriShellPiece, QuadShellPiece, HexagonShellPiece etc.  Currently
   provides implementation for Rectangular Shell Piece.

*/
/////////////////////////////////////////////////////////////////////////////
	
  class PlaneShellPiece : public ShellGeometryPiece {
	 
  public:
    //////////
    //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
    // input specification and builds a plane.
    PlaneShellPiece(ProblemSpecP &);
	 
    //////////
    // Destructor
    virtual ~PlaneShellPiece();

    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const;
	 
    //////////
    // Determines whether a point is inside the plane. 
    virtual bool inside(const Point &p) const;
	 
    //////////
    // Returns the bounding box surrounding the box.
    virtual Box getBoundingBox() const;

    //////////
    // Returns the number of particles
    int returnParticleCount(const Patch* patch);

    //////////
    // Creates the particles
    int createParticles(const Patch* patch,
			ParticleVariable<Point>&  pos,
			ParticleVariable<double>& vol,
			ParticleVariable<double>& pThickTop,
			ParticleVariable<double>& pThickBot,
			ParticleVariable<Vector>& pNormal,
			ParticleVariable<Vector>& psize,
			particleIndex start);


  private:
    virtual void outputHelper( ProblemSpecP & ps ) const;

    Point  d_center;
    Vector d_normal;
    double d_radius;
    double d_thickness;
    int d_numRadius;
  };
} // End namespace Uintah

#endif // __PLANE_SHELL_PIECE_H__
