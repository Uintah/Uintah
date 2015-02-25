/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __FILE_GEOMETRY_PIECE_H__
#define __FILE_GEOMETRY_PIECE_H__

#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Geometry/Point.h>

#include   <vector>
#include   <list>
#include   <string>

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
	
  Reads in a set of points (particle centroids) from an input file.  

  In addition, Convected Particle Domain Interpolation (CPDI) or the 
  Convected Particle domain Triangle/Tetrahedral Interpolation (CPTI)
  particle domain descriptions can be read if the vectors rvec1, rvec2
  and rvec3 are specified.  The results are stored in the columns of 
  the Size matrix.  Contact Brian Leavy for more information.

  The input form looks like this:
  \verbatim
    <file>
      <name>file_name.pts</name>
      <format>text </format>
      <var>p.volume</var>
      <var>p.fiberdir</var>
      <var>p.externalforce</var>
      <var>p.rvec1</var>         <!-- CPDI or CPTI -->
      <var>p.rvec2</var>         <!-- CPDI or CPTI -->
      <var>p.rvec3</var>         <!-- CPDI or CPTI -->
    </file>
  \endverbatim
  
  Requires one input: file name <name>filename.pts</name>
  
  The format field can be used to specify that the point file is 
    text  - plain text list of points (slow for may processors)
    lsb   - least significant byte binary double
    msb   - most significant byte binary double
    bin   - use native binary ordering.
    
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
    FileGeometryPiece(const std::string& file_name);
    
    //////////
    // Destructor
    virtual ~FileGeometryPiece();

    static const std::string TYPE_NAME;
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
 
    Box                    d_box;
    std::string            d_file_name;
    std::string            d_file_format;
    std::list<std::string> d_vars;
    bool                   d_usePFS;
    
    void checkFileType(std::ifstream & source, std::string& fileType, std::string& filename);
    
    bool read_line(std::istream & is, Point & xmin, Point & xmax);
    void read_bbox(std::istream & source, Point & lowpt, Point & highpt) const;
    virtual void outputHelper( ProblemSpecP & ps ) const;
  };
  
} // End namespace Uintah

#endif // __FILE_GEOMETRY_PIECE_H__
