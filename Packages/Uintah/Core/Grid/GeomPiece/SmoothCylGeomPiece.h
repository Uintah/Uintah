#ifndef __SMOOTH_CYL_PIECE_H__
#define __SMOOTH_CYL_PIECE_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/SmoothGeomPiece.h>
#include <Core/Geometry/Point.h>

#include <math.h>
#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/////////////////////////////////////////////////////////////////////////////
/*!
	
 \class SmoothCylGeomPiece
	
 \brief Creates a smooth cylinder with end-caps
	
 \warning Does not allow for correct application of symmetry 
          boundary conditions.  Use symmetry at your own risk.
          The end caps are exactly the same diameter as the outer
          diameter of the cylinder and are welded perfectly to the 
          cylinder.

 \author  Biswajit Banerjee \n
          C-SAFE and Department of Mechanical Engineering \n
          University of Utah \n

   Creates a smooth solid/hollow cylinder with/without end-caps from the 
   xml input 
   file description.
   The input form for a solid cylinder looks like this: \n
   \verbatim
   <smoothcyl> 
     <bottom>[0.0,0.0,0.0]</bottom> 
     <top>[0.0,0.0,10.0]</top> 
     <radius>2.0</radius> 
     <num_radial>20</num_radial> 
     <num_axial>100</num_axial> 
   </smoothcyl> 
   \endverbatim
   The input form for a hollow cylinder with end-caps looks like this: \n
   \verbatim
   <smoothcyl> 
     <bottom>[0.0,0.0,0.0]</bottom> 
     <top>[0.0,0.0,10.0]</top> 
     <radius>2.0</radius> 
     <thickness>0.1</thickness> 
     <endcap_thickness>1.0</endcap_thickness> 
     <num_radial>20</num_radial> 
     <num_axial>100</num_axial> 
   </smoothcyl> 
   \endverbatim
   If the points are to be written to an output file, use the following
   \verbatim
   <smoothcyl> 
     <bottom>[0.0,0.0,0.0]</bottom> 
     <top>[0.0,0.0,10.0]</top> 
     <radius>2.0</radius> 
     <thickness>0.1</thickness> 
     <endcap_thickness>1.0</endcap_thickness> 
     <num_radial>20</num_radial> 
     <num_axial>100</num_axial> 
     <output_file>"fileName"</output_file>
   </smoothcyl> 
   \endverbatim
	
*/
/////////////////////////////////////////////////////////////////////////////

  class SmoothCylGeomPiece : public SmoothGeomPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*!  
      \brief Constructor that takes a ProblemSpecP argument.   
      It reads the xml input specification and builds a cylinder.
    */
    //////////////////////////////////////////////////////////////////////
    SmoothCylGeomPiece(ProblemSpecP &);
	 
    //////////////////////////////////////////////////////////////////////
    /*! Destructor */
    //////////////////////////////////////////////////////////////////////
    virtual ~SmoothCylGeomPiece();
	 
    //////////////////////////////////////////////////////////////////////
    /*! Determines whether a point is inside the cylinder. */
    //////////////////////////////////////////////////////////////////////
    virtual bool inside(const Point &p) const;
	 
    //////////////////////////////////////////////////////////////////////
    /*! Returns the bounding box surrounding the box. */
    //////////////////////////////////////////////////////////////////////
    virtual Box getBoundingBox() const;

  protected:

    //////////////////////////////////////////////////////////////////////
    /*! Creates the particles */
    //////////////////////////////////////////////////////////////////////
    int createPoints();

    //////////////////////////////////////////////////////////////////////
    /*! Creates the particles for the two end caps */
    //////////////////////////////////////////////////////////////////////
    int createEndCapPoints();

    //////////////////////////////////////////////////////////////////////
    /*! Creates the particles for the solid cylinder */
    //////////////////////////////////////////////////////////////////////
    int createSolidCylPoints();

    //////////////////////////////////////////////////////////////////////
    /*! Creates the particles for the hollow cylinder */
    //////////////////////////////////////////////////////////////////////
    int createHollowCylPoints();

  private:
	 
    Point  d_top;
    Point  d_bottom;
    double d_radius;
    double d_thickness;
    double d_capThick;
    int d_numRadial;
    int d_numAxial;

  };
} // End namespace Uintah

#endif // __SMOOTH_CYL_PIECE_H__
