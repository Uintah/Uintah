#ifndef __FILE_GEOMETRY_OBJECT_H__
#define __FILE_GEOMETRY_OBJECT_H__

#include <Packages/Uintah/Core/GeometryPiece/SmoothGeomPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Core/Geometry/Point.h>

#include <sgi_stl_warnings_off.h>
#include   <vector>
#include   <list>
#include   <string>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/GeometryPiece/share.h>
namespace Uintah {

/////////////////////////////////////////////////////////////////////////////
/*!
	
  \class FileGeometryPiece
	
  \brief Reads in a set of points and optionally volume, external forces and 
  fiber directions for each point from an input text file.
  
  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n
	
  Reads in a set of points from an input file.  

  The input form looks like this:
  \verbatim
    <file>
      <name>file_name.txt</name>
      <format>text </format>
      <split> true </split>
      <var>p.volume</var>
      <var>p.fiberdir</var>
      <var>p.externalforce</var>
    </file>
  \endverbatim
  
  Requires one input: file name <name>points.txt</name>
  
  The format field can be used to specify that the point file is 
    text  - plain text list of points (slow for may processors)
    lsb   - least significant byte binary double
    msb   - most significant byte binary double
    bin   - use native binary ordering.

    if split is specified (the default), the points file is assumed to have
    been run through pfs.
    
    the 'split' format must match the number of processors, and 
    expects files in file_name.txt.iproc
    where iproc is the mpi rank of the processor.
    
    Note, for all formats (text and binary), there needs to be a 128 line
    buffer containing the bounding box of the whole data set in every file.
  
  If <var?> tags are present, extra fields values can be assigned to each 
  point.

  the order of the var field determines the expected column order of 
  the field; one column for volume and three for force and direction.
  
  There are methods for checking if a point is inside the box
  and also for determining the bounding box for the box (which
  just returns the box itself).
  
*/
/////////////////////////////////////////////////////////////////////////////
	
  using std::vector;
  using std::string;
  using std::list;

  class SCISHARE FileGeometryPiece : public SmoothGeomPiece {
    
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

    static const string TYPE_NAME;
    virtual std::string getType() const { return TYPE_NAME; }

    /// Make a clone
    virtual GeometryPieceP clone() const;

    //////////
    // Determines whether a point is inside the box.
    virtual bool inside(const Point &p) const;
	 
    //////////
    //  Returns the bounding box surrounding the cylinder.
    virtual Box getBoundingBox() const;

    void readPoints(int pid);

    unsigned int createPoints();

  private:
 
    Box                 d_box;
    string              d_file_name;
    bool                d_presplit;
    string              d_file_format;
    list<string>        d_vars;
    
    bool read_line(std::istream & is, Point & xmin, Point & xmax);
    void read_bbox(std::istream & source, Point & lowpt, Point & highpt) const;
    virtual void outputHelper( ProblemSpecP & ps ) const;
  };
  
} // End namespace Uintah

#endif // __FILE_GEOMTRY_Piece_H__
