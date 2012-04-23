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


#ifndef __SHELL_GEOMETRY_OBJECT_H__
#define __SHELL_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <cmath>


namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ShellGeometryPiece
	
    \brief Abstract class for shell geometries
	
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
	
    Creates a shell from the xml input file description.
    The input form looks like this:
   
    \verbatim
    a) Sphere
      <sphere_shell>
        <origin>[0.,0.,0.]</origin>
        <radius>2.0</radius>
        <thickness>0.1</thickness>
        <num_lat>20</num_lat>
        <num_long>40</num_long>
      </sphere_shell>

    b) Cylinder
      <cylinder_shell>
      </cylinder_shell>
	
    c) Plane
      <plane_shell>
      </plane_shell>

    \endverbatim
	
  */
  /////////////////////////////////////////////////////////////////////////////

  class ShellGeometryPiece : public GeometryPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*! \brief Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~ShellGeometryPiece();

    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const = 0;
	 
    //////////////////////////////////////////////////////////////////////
    /*! \brief Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const = 0;

    //////////////////////////////////////////////////////////////////////
    /*! \brief Determines whether a point is inside the shell */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const = 0;
	 
    //////////////////////////////////////////////////////////////////////
    /*! \brief Returns the number of particles associated with the shell */
    //////////////////////////////////////////////////////////////////////
    virtual int returnParticleCount(const Patch* patch) = 0;

    //////////////////////////////////////////////////////////////////////
    /*! \brief Create the particles in the shell */
    //////////////////////////////////////////////////////////////////////
    virtual int createParticles(const Patch* patch,
				ParticleVariable<Point>&  pos,
				ParticleVariable<double>& vol,
				ParticleVariable<double>& pThickTop,
				ParticleVariable<double>& pThickBot,
				ParticleVariable<Vector>& pNormal,
				ParticleVariable<Vector>& psize,
				particleIndex start) = 0;

  protected:
    virtual void outputHelper( ProblemSpecP & ps ) const = 0;

    ShellGeometryPiece();

  };
} // End namespace Uintah

#endif // __SHELL_GEOMETRY_PIECE_H__
