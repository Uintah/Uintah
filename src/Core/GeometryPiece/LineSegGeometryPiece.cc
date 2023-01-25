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

#include <Core/GeometryPiece/LineSegGeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Box.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/Plane.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;

#define INSIDE_NEW

const string LineSegGeometryPiece::TYPE_NAME = "line_segment";
//______________________________________________________________________
//

LineSegGeometryPiece::LineSegGeometryPiece(ProblemSpecP &ps)
{
  name_ = "Unnamed LineSeg";

  ps->require("name",d_file);

  readPoints(d_file);

  d_points.clear();
}

//______________________________________________________________________
//

LineSegGeometryPiece::LineSegGeometryPiece(string filename)
{
  name_ = "Unnamed LineSeg";

  d_file = filename;

  readPoints(d_file);
}

//______________________________________________________________________
//

LineSegGeometryPiece::LineSegGeometryPiece(const LineSegGeometryPiece& copy)
{
  d_box    = copy.d_box;
  d_points = copy.d_points;
  d_LS     = copy.d_LS;
}

//______________________________________________________________________
//

LineSegGeometryPiece& LineSegGeometryPiece::operator=(const LineSegGeometryPiece& rhs)
{
  if (this == &rhs) {
    return *this;
  }
  
  //__________________________________
  // Clean out lhs
  d_points.clear();
  d_LS.clear();

  // Copy the rhs stuff
  d_box    = rhs.d_box;
  d_points = rhs.d_points;
  d_LS     = rhs.d_LS;

  return *this;
}

//______________________________________________________________________
//

LineSegGeometryPiece::~LineSegGeometryPiece()
{
  d_points.clear();
  d_LS.clear();
}

//______________________________________________________________________
//

void
LineSegGeometryPiece::outputHelper( ProblemSpecP & ps ) const
{
  ps->appendElement("name",d_file);
}

//______________________________________________________________________
//

GeometryPieceP
LineSegGeometryPiece::clone() const
{
  return scinew LineSegGeometryPiece(*this);
}

//______________________________________________________________________
//
bool
LineSegGeometryPiece::inside(const Point &p,
                             const bool useNewestVersion=false) const
{
  // Count the number of times a ray from the point p
  // intersects the line segment surface.  If the number
  // of crossings is odd, the point is inside, else it
  // is outside.

  // Check if Point p is outside the bounding box
  if (!(p == Max(p,d_box.lower()) && p == Min(p,d_box.upper()))) {
    return false;
  }
  
  // Cast a ray from point P to some point at "infinity".
  // Loop over the line segments, count how many of the segments the
  // ray intersects with
  Vector rayDirection1(1.0/M_PI, M_PI, 0.0);
  rayDirection1/=rayDirection1.length();
  int count1 = 0;
  for (unsigned int i = 0; i < d_LS.size(); i++) {
     LineSeg tmp = d_LS[i]; 
     Point point1 = tmp.vertex(0);
     Point point2 = tmp.vertex(1);
     Vector v1 = p - point1;
     Vector v2 = point2 - point1;
     Vector v3 = Vector(-rayDirection1.y(), rayDirection1.x(), 0.0);

     double dot = Dot(v2,v3);
     if(! (fabs(dot) < 1.e-10)){
       double t1 = Cross(v2, v1).z()/dot;
       double t2 = Dot(v1,v3)/dot;

       if (t1 >= 0.0 && (t2 >= 0.0 && t2 <= 1.0)){
         count1++;
       }
     }
  }

  if(count1%2==0){
    return false;
  } else {
    return true;
  }
}

//______________________________________________________________________
//

Box
LineSegGeometryPiece::getBoundingBox() const
{
  return d_box;
}

//______________________________________________________________________
//
void
LineSegGeometryPiece::readPoints(const string& file)
{
  string f = file;
  std::ifstream source(f.c_str());
  
  if (!source) {
    std::ostringstream warn;
    warn << "\n ERROR: opening geometry pts points file ("<< f
         << ").\n  The file must be in the same directory as sus \n"
         << "  Do not enclose the filename in quotation marks\n";
    throw ProblemSetupException(warn.str(),__FILE__, __LINE__);
  }

  double x,y;
  while (source >> x >> y) {
    d_points.push_back(Point(x,y,0));
  }

  source.close();

  // Create LineSegments
  for(unsigned int i = 0; i < d_points.size()-1; i++){
     LineSeg tmpLS = LineSeg(d_points[i], d_points[i+1]);
     d_LS.push_back(tmpLS);
  }
  LineSeg tmpLS = LineSeg(d_points[d_points.size()-1], d_points[0]);
  d_LS.push_back(tmpLS);

  // Find the min and max points so that the bounding box can be determined.
  Point min( 1e30, 1e30, 1e30);
  Point max(-1e30,-1e30,-1e30);
  
  vector<Point>::const_iterator itr;
  for (itr = d_points.begin(); itr != d_points.end(); ++itr) {
    min = Min(*itr,min);
    max = Max(*itr,max);
  }
  Vector fudge(1.e-5,1.e-5,1.e-5);
  min = min - fudge;
  max = max + fudge;
  d_box = Box(min,max);
}
