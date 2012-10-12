/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __GUV_PIECE_H__
#define __GUV_PIECE_H__

#include <Core/GeometryPiece/ShellGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class GUVSphereShellPiece
        
    \brief Creates a GUV from the xml input file description.
        
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    2004 University of Utah \n
        
    Creates a GUV sphere from the xml input file description.
    The input form looks like this:
    \verbatim
    <GUV_sphere>
      <origin>[0.,0.,0.]</origin>
      <radius>2.0</radius>
      <thickness_lipid>0.1</thickness_lipid>
      <thickness_cholesterol>0.1</thickness_cholesterol>
      <zone name="zone1">
        <zone_center_theta> 30 </zone_center_theta>
        <zone_center_phi> 60 </zone_center_phi>
        <zone_radius> 0.2 </zone_radius>
      </zone>
      <zone name="zone2">
        <zone_center_theta> -10 </zone_center_theta>
        <zone_center_phi> -20 </zone_center_phi>
        <zone_radius> 0.3 </zone_radius>
      </zone>
    </GUV_sphere>
    \endverbatim
        
  */
  /////////////////////////////////////////////////////////////////////////////

  class GUVSphereShellPiece : public ShellGeometryPiece {
         
  public:
    //////////
    //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
    // input specification and builds a sphere.
    GUVSphereShellPiece(ProblemSpecP &);
         
    //////////
    // Destructor
    virtual ~GUVSphereShellPiece();

    static const string TYPE_NAME;
    std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const;
         
    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside the cylinder. */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const;
         
    //////////////////////////////////////////////////////////////////////
    /*! Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const;

    //////////////////////////////////////////////////////////////////////
    /*! Creates points and returns count of points */
    //////////////////////////////////////////////////////////////////////
    int createPoints();

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle locations */
    //////////////////////////////////////////////////////////////////////
    inline vector<Point>* get_position()
    {
      return &d_pos;
    }

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle volumes */
    //////////////////////////////////////////////////////////////////////
    inline vector<double>* getVolume()
    {
      return &d_vol;
    }

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle types */
    //////////////////////////////////////////////////////////////////////
    inline vector<int>* getType()
    {
      return &d_type;
    }

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle thickness */
    //////////////////////////////////////////////////////////////////////
    inline vector<double>* getThickness()
    {
      return &d_thick;
    }

    //////////////////////////////////////////////////////////////////////
    /*! Returns the vector containing the set of particle normals */
    //////////////////////////////////////////////////////////////////////
    inline vector<Vector>* getNormal()
    {
      return &d_norm;
    }

    //////////////////////////////////////////////////////////////////////
    /*! Returns the number of particles */
    //////////////////////////////////////////////////////////////////////
    inline int returnPointCount() const
    {
      return d_pos.size();
    }

    //////////////////////////////////////////////////////////////////////
    /*! Set the particle spacing */
    //////////////////////////////////////////////////////////////////////
    inline void setParticleSpacing(double dx)
    {
      d_dx = dx;
    }

  protected:

    virtual void outputHelper( ProblemSpecP & ps ) const;

    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside a cholesterol zone. */
    //////////////////////////////////////////////////////////////////////
    bool insideZone(const Point &p) const;
         
    vector<Point>  d_pos;
    vector<double> d_vol;
    vector<int>    d_type;
    vector<double> d_thick;
    vector<Vector> d_norm;
    double         d_dx;

  private:
  
    int returnParticleCount(const Patch* patch);
    int createParticles(const Patch* patch,
			ParticleVariable<Point>&  pos,
			ParticleVariable<double>& vol,
			ParticleVariable<double>& pThickTop,
			ParticleVariable<double>& pThickBot,
			ParticleVariable<Vector>& pNormal,
			ParticleVariable<Matrix3>& psize,
			particleIndex start);

    Point  d_origin;
    double d_radius;
    double d_h_lipid;
    double d_h_cholesterol;
    vector<double> d_theta_zone;
    vector<double> d_phi_zone;
    vector<double> d_radius_zone;
  };
} // End namespace Uintah

#endif // __GUV_PIECE_H__
