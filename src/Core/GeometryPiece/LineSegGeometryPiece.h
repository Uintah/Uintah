/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef __LS_GEOMETRY_OBJECT_H__
#define __LS_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/UniformGrid.h>
#include <Core/Grid/Box.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Plane.h>

#include   <vector>

namespace Uintah {


/**************************************

CLASS
   LineSegGeometryPiece

   Creates geometry based on a sequential set of x-y points.

GENERAL INFORMATION

   LineSegGeometryPiece.h

   James Guilkey 
   Department of Mechanical Engineering
   University of Utah

KEYWORDS
   LineSegGeometryPiece BoundingBox inside

DESCRIPTION
   Fill geometry described by a file containing a sequential set of x-y points.
   This obviously only works for 2-D and assumes that the points are in the
   x-y plane (z=0).
   Requires one input: file name containing the points
   There are methods for checking if a point is inside the surface
   and also for determining the bounding box for the surface.
   The input form looks like this:
       <line_segment>
         <file>surface.pts</file>
       </line_segment>


WARNING

****************************************/

      class LineSegGeometryPiece : public GeometryPiece {
      public:
         //////////
         //  Constructor that takes a ProblemSpecP argument.   It reads the xml
         // input specification and builds the triangulated surface piece.
         LineSegGeometryPiece(ProblemSpecP &);
         //////////
         LineSegGeometryPiece(std::string filename);

         LineSegGeometryPiece(const LineSegGeometryPiece&);

         virtual ~LineSegGeometryPiece();
         
         LineSegGeometryPiece& operator=(const LineSegGeometryPiece&);

         static const std::string TYPE_NAME;
         virtual std::string getType() const { return TYPE_NAME; }

         virtual GeometryPieceP clone() const;

         //////////
         // Determins whether a point is inside the triangulated surface.
         virtual bool inside(const Point &p, const bool defaultValue) const;
         
         //////////
         // Returns the bounding box surrounding the triangulated surface.
         virtual Box getBoundingBox() const;

         void scale(const double factor);

         double surfaceArea() const;

      private:

         virtual void outputHelper( ProblemSpecP & ps ) const;

         void readPoints(const std::string& file);

         std::string d_file;
         Box d_box;
         std::vector<Point>       d_points;
         std::vector<LineSeg> d_LS;
      };

} // End namespace Uintah

#endif // __LS_GEOMETRY_PIECE_H__
