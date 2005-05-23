#ifndef __CORRUG_EDGE_PIECE_H__
#define __CORRUG_EDGE_PIECE_H__

#include <Packages/Uintah/Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Geometry/Point.h>

#include <math.h>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/////////////////////////////////////////////////////////////////////////////
/*!
	
 \class CorrugEdgeGeomPiece
	
 \brief Creates a plate with one edge corrugated. 
	
 \warning Does not allow for correct application of symmetry 
          boundary conditions.  Use symmetry at your own risk.
          Normal is not used .. needs to be implemented.
          Volume is not calculated accurately.

 \author  Biswajit Banerjee \n
          C-SAFE and Department of Mechanical Engineering \n
          University of Utah \n

   Creates a plate with one corrugated edge.  The particle spacing
   is determined from the grid size and the number of particles
   per grid cell.\n
   The input form for a solid cylinder looks like this: \n
   \verbatim
   <corrugated> 
     <xymin>      [0.0,0.0,0.0]    </xymin> 
     <xymax>      [20.0,20.0,0.0]  </xymax> 
     <thickness>  1.0              </thickness>
     <normal>     [0.0,0.0,1.0]    </normal>
     <corr_edge>  x+               </corr_edge>
     <curve>      sin              </curve>
     <wavelength> 2.0              </wavelength>
     <amplitude>  2.0              </amplitude>
   </corrugated> 
   \endverbatim
*/
/////////////////////////////////////////////////////////////////////////////

  class CorrugEdgeGeomPiece : public SmoothGeomPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*!  
      \brief Constructor that takes a ProblemSpecP argument.   
      It reads the xml input specification and builds a plate with
      one corrugated edge.
    */
    //////////////////////////////////////////////////////////////////////
    CorrugEdgeGeomPiece(ProblemSpecP &);
	 
    //////////////////////////////////////////////////////////////////////
    /*! Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~CorrugEdgeGeomPiece();

    /// Make a clone
    CorrugEdgeGeomPiece* clone();
	 
    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside the cylinder. */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const;
	 
    //////////////////////////////////////////////////////////////////////
    /*! Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const;

    //////////////////////////////////////////////////////////////////////
    /*! Creates the particles */
    //////////////////////////////////////////////////////////////////////
    int createPoints();

  private:
	 
    Point  d_xymin;
    Point  d_xymax;
    double d_thickness;
    Vector d_normal;
    string d_edge;
    string d_curve;
    double d_wavelength;
    double d_amplitude;
  };
} // End namespace Uintah

#endif // __CORRUG_EDGE_PIECE_H__
