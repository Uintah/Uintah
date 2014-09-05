#ifndef __FILE_GEOMETRY_OBJECT_H__
#define __FILE_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/Grid/GeomPiece/SmoothGeomPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

/////////////////////////////////////////////////////////////////////////////
/*!
	
  \class FileGeometryPiece
	
  \brief Reads in a set of points and optionally a volume for each point 
  from an input text file.
	
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n
	
  Reads in a set of points from an input file.  Optionally, if the
  <var> tag is present, the volume will be read in for each point set.
  Requires one input: file name <name>points.txt</name>
  Optional input : <var1>p.volume </var1>
  There are methods for checking if a point is inside the box
  and also for determining the bounding box for the box (which
  just returns the box itself).
  The input form looks like this:
  \verbatim
    <name>file_name.txt</name>
      <var1>p.volume</var1>
      <var2>p.externalforce</var2>
      <var3>p.fiberdir</var3>
  \endverbatim
	
*/
/////////////////////////////////////////////////////////////////////////////
	
  using std::vector;

  class FileGeometryPiece : public SmoothGeomPiece {
	 
  public:
    //////////////////////////////////////////////////////////////////////
    /*! \brief Constructor that takes a ProblemSpecP argument.   
        It reads the xml input specification and builds a generalized box. */
    //////////////////////////////////////////////////////////////////////
    FileGeometryPiece(ProblemSpecP&);

    //////////////////////////////////////////////////////////////////////
    /*! Construct a box from a min/max point */
    //////////////////////////////////////////////////////////////////////
    FileGeometryPiece(const string& file_name);
	 
    //////////
    // Destructor
    virtual ~FileGeometryPiece();
	 
    /// Make a clone
    FileGeometryPiece* clone();

    //////////
    // Determines whether a point is inside the box.
    virtual bool inside(const Point &p) const;
	 
    //////////
    //  Returns the bounding box surrounding the cylinder.
    virtual Box getBoundingBox() const;

    void readPoints(int pid);

    int createPoints();

  private:
    Box d_box;
    bool d_var1_bool;
    bool d_var2_bool;
    bool d_var3_bool;
    string d_file_name,d_var_name1,d_var_name2,d_var_name3;
  };
} // End namespace Uintah

#endif // __FILE_GEOMTRY_Piece_H__
