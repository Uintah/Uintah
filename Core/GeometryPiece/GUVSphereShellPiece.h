#ifndef __GUV_PIECE_H__
#define __GUV_PIECE_H__

#include <Packages/Uintah/Core/GeometryPiece/ShellGeometryPiece.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>

#include <math.h>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

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

    /// Make a clone
    GUVSphereShellPiece* clone();
         
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
    inline vector<Point>* getPosition()
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

    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside a cholesterol zone. */
    //////////////////////////////////////////////////////////////////////
    bool insideZone(const Point &p) const;
         
    vector<Point> d_pos;
    vector<double> d_vol;
    vector<int> d_type;
    vector<double> d_thick;
    vector<Vector> d_norm;
    double d_dx;

  private:
  
    int returnParticleCount(const Patch* patch);
    int createParticles(const Patch* patch,
			ParticleVariable<Point>&  pos,
			ParticleVariable<double>& vol,
			ParticleVariable<double>& pThickTop,
			ParticleVariable<double>& pThickBot,
			ParticleVariable<Vector>& pNormal,
			ParticleVariable<Vector>& psize,
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
