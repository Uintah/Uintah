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

#ifndef __TRI_GEOMETRY_OBJECT_H__
#define __TRI_GEOMETRY_OBJECT_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/UniformGrid.h>
#include <Core/Grid/Box.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Plane.h>

#include   <vector>

namespace Uintah {

using std::vector;
using namespace SCIRun;

/**************************************
        
CLASS
   TriGeometryPiece
        
   Creates a triangulated surface piece from the xml input file description.
        
GENERAL INFORMATION
        
   TriGeometryPiece.h
        
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
        
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
        
 
        
KEYWORDS
   TriGeometryPiece BoundingBox inside
        
DESCRIPTION
   Creates a triangulated surface piece from the xml input file description.
   Requires one input: file name (convetion use suffix .dat).  
   There are methods for checking if a point is inside the surface
   and also for determining the bounding box for the surface.
   The input form looks like this:
       <tri>
         <file>surface.dat</file>
       </tri>
        
        
WARNING
        
****************************************/

      class TriGeometryPiece : public GeometryPiece {
      public:
         //////////
         //  Constructor that takes a ProblemSpecP argument.   It reads the xml 
         // input specification and builds the triangulated surface piece.
         TriGeometryPiece(ProblemSpecP &);
         //////////

         TriGeometryPiece(const TriGeometryPiece&);

         TriGeometryPiece& operator=(const TriGeometryPiece&);
         
         // Destructor
         virtual ~TriGeometryPiece();

         static const string TYPE_NAME;
         virtual std::string getType() const { return TYPE_NAME; }

         virtual GeometryPieceP clone() const;

         //////////
         // Determins whether a point is inside the triangulated surface.
         virtual bool inside(const Point &p) const;
         bool insideNew(const Point &p, int& cross) const;
         
         //////////
         // Returns the bounding box surrounding the triangulated surface.
         virtual Box getBoundingBox() const;

         void scale(const double factor);

         double surfaceArea() const;
         
      private:

         virtual void outputHelper( ProblemSpecP & ps ) const;
         
         void readPoints(const string& file);
         void readTri(const string& file);
         void makePlanes();
         void makeTriBoxes();
         void insideTriangle(Point& p, int i, int& NCS, int& NES) const;
         
         std::string d_file;
         Box d_box;
         vector<Point>     d_points;
         vector<IntVector> d_tri;
         vector<Plane>     d_planes;
         vector<Box>       d_boxes;

         UniformGrid* d_grid;
         
      };

} // End namespace Uintah

#endif // __TRI_GEOMETRY_PIECE_H__
