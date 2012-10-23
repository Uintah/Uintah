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
 *  BBox.h: Bounding box class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   1994
 *
 */

#ifndef Geometry_BBox_h
#define Geometry_BBox_h

#include <Core/Geometry/Point.h>

#include <Core/Geometry/share.h>

#include   <ostream>

// some compilers define the min and max macro, and the BBox::min/max will get confused.
#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif

namespace SCIRun {

  class Vector;
  class Piostream;

  class SCISHARE BBox {
    
  protected:
    SCISHARE friend void Pio( Piostream &, BBox& );

  public:
    BBox();
    ~BBox();
    BBox(const BBox&);
    BBox& operator=(const BBox&);
    BBox(const Point& min, const Point& max);
    inline int valid() const {return is_valid;}
    void reset();

    // Expand the bounding box to include point p
    void extend(const Point& p);

    // Expand the bounding box to include a sphere of radius radius
    // and centered at point p
    void extend(const Point& p, double radius);

    // Expand the bounding box to include bounding box b
    void extend(const BBox& b);

    // Expand the bounding box to include a disc centered at cen,
    // with normal normal, and radius r.
    void extend_disc(const Point& cen, const Vector& normal, double r);

    Point center() const;
    double longest_edge();

    // Move the bounding box 
    void translate(const Vector &v);

    // Scale the bounding box by s, centered around o
    void scale(double s, const Vector &o);

    Point min() const;
    Point max() const;
    Vector diagonal() const;

    inline bool inside(const Point &p) const {return (is_valid && p.x()>=cmin.x() && p.y()>=cmin.y() && p.z()>=cmin.z() && p.x()<=cmax.x() && p.y()<=cmax.y() && p.z()<=cmax.z());}

    bool overlaps( const BBox& bb );
    bool overlaps2( const BBox& bb );

    // returns true if the ray hit the bbox and returns the hit point
    // in hitNear
    bool intersect( const Point& e, const Vector& v, Point& hitNear );

    friend std::ostream& operator<<(std::ostream& out, const BBox& b);
 
    bool operator==( const BBox &b ) const{ return cmin==b.cmin && cmax==b.cmax;};
    bool operator!=( const BBox &b ) const{ return cmin!=b.cmin || cmax!=b.cmax;};
  private:
    Point cmin;
    Point cmax;
    bool is_valid;
  };
} // End namespace SCIRun


#endif
