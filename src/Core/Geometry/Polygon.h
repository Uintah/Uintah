/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


/*
 *  Polygon.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 */

#ifndef Geometry_Polygon_h
#define Geometry_Polygon_h 1

#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <iosfwd>
#include <vector>
#include <string>

namespace SCIRun {
    
using std::vector;

/**************************************

 CLASS
 Polygon

 Simple Polygon class

 GENERAL INFORMATION

 Polygon.h

 Kurt Zimmerman
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

 
 KEYWORDS
 Polygon

 DESCRIPTION
 Simple Polygon class. 

 WARNING
 This class does not check to see if 
 indeed the points are coplanar.  To be added later.
****************************************/


class Polygon {
  vector<Point> vertices;
  vector<Point> texcoords;
public:

  // GROUP:  Constructors:
  //////////
  // Constructor
  // parameters: a pointer to a C array of points that correspond
  //             to the vertices and an int specifying the length 
  //             of the array.  Points must be placed in counter
  //             clockwise order.
  Polygon(const Point *p, const Point *t, int nPoints);

  //////////
  // Constructor
  // takes a std:vector of Points that are a counter clockwise
  // ordering of the vertices.
  Polygon(const vector<Point>& v, const vector<Point>& t);
  //////////
  // Copy Constructor
  Polygon(const Polygon&);
  
  // GROUP: Destructors
  //////////
  // Destructor
  ~Polygon(){};

  // GROUP: Boolean operators
  //////////
  // bool
  bool operator==(const Polygon&) const;
  //////////
  // bool
  bool operator!=(const Polygon&) const;

  Polygon& operator=(const Polygon&);

  // GROUP: info/access
  //////////
  // index method: return the ith vertex
  const Point& operator[](int i) const; 
  //////////
  // index method: return the ith vertex
  const Point& getVertex(int i) const; 
  //////////
  // index method: return the ith texture coordinate;
  const Point& getTexCoord(int i) const; 
  //////////
  // size method: How long is the vector?
  int size() const { return int(vertices.size()); }

  //////////
  // string method:
  string get_string() const;

private:
  //////////
  // Hidden Constructor
  Polygon(){};
};

std::ostream& operator<<(std::ostream& os, const Polygon& p);

} // End namespace SCIRun

#endif //ifndef Geometry_Polygon_h






