//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : RectRegion.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:05:14 2006

#include <Core/Skinner/RectRegion.h>
#include <Core/Math/MinMax.h>


namespace SCIRun {
  namespace Skinner {
    RectRegion::RectRegion()
    {
      double x1 = AIR_POS_INF;
      double y1 = AIR_POS_INF;
      double x2 = AIR_NEG_INF;
      double y2 = AIR_NEG_INF;

      coords_[0] = x1;
      coords_[1] = y1;
      coords_[2] = x2;
      coords_[3] = y2;

      coords_2D_[0] = x1;
      coords_2D_[1] = y1;
      coords_2D_[2] = x2;
      coords_2D_[3] = y1;
      coords_2D_[4] = x2;
      coords_2D_[5] = y2;
      coords_2D_[6] = x1;
      coords_2D_[7] = y2;

      coords_3D_[0] = x1;
      coords_3D_[1] = y1;
      coords_3D_[2] = 0.0;
      coords_3D_[3] = x2;
      coords_3D_[4] = y1;
      coords_3D_[5] = 0.0;
      coords_3D_[6] = x2;
      coords_3D_[7] = y2;
      coords_3D_[8] = 0.0;
      coords_3D_[9] = x1;
      coords_3D_[10] = y2;
      coords_3D_[11] = 0.0;
    }

    RectRegion::RectRegion(double x1, double y1, double x2, double y2)
    {
      coords_[0] = x1;
      coords_[1] = y1;
      coords_[2] = x2;
      coords_[3] = y2;

      coords_2D_[0] = x1;
      coords_2D_[1] = y1;
      coords_2D_[2] = x2;
      coords_2D_[3] = y1;
      coords_2D_[4] = x2;
      coords_2D_[5] = y2;
      coords_2D_[6] = x1;
      coords_2D_[7] = y2;

      coords_3D_[0] = x1;
      coords_3D_[1] = y1;
      coords_3D_[2] = 0.0;
      coords_3D_[3] = x2;
      coords_3D_[4] = y1;
      coords_3D_[5] = 0.0;
      coords_3D_[6] = x2;
      coords_3D_[7] = y2;
      coords_3D_[8] = 0.0;
      coords_3D_[9] = x1;
      coords_3D_[10] = y2;
      coords_3D_[11] = 0.0;
    }

    RectRegion::RectRegion(const RectRegion &region)
    { 
      coords_[0] = region.x1();
      coords_[1] = region.y1();
      coords_[2] = region.x2();
      coords_[3] = region.y2();

      coords_2D_[0] = region.x1();
      coords_2D_[1] = region.y1();
      coords_2D_[2] = region.x2();
      coords_2D_[3] = region.y1();
      coords_2D_[4] = region.x2();
      coords_2D_[5] = region.y2();
      coords_2D_[6] = region.x1();
      coords_2D_[7] = region.y2();

      coords_3D_[0] = region.x1();
      coords_3D_[1] = region.y1();
      coords_3D_[2] = 0.0;
      coords_3D_[3] = region.x2();
      coords_3D_[4] = region.y1();
      coords_3D_[5] = 0.0;
      coords_3D_[6] = region.x2();
      coords_3D_[7] = region.y2();
      coords_3D_[8] = 0.0;
      coords_3D_[9] = region.x1();
      coords_3D_[10] = region.y2();
      coords_3D_[11] = 0.0;
    }

    double
    RectRegion::x1() const {
      return coords_[0];
    }

    double
    RectRegion::y1() const {
      return coords_[1];
    }

    double
    RectRegion::x2() const {
      return coords_[2];
    }

    double
    RectRegion::y2() const {
      return coords_[3];
    }

    double
    RectRegion::width() const {
      return coords_[2]-coords_[0];
    }

    double
    RectRegion::height() const {
      return coords_[3]-coords_[1];
    }

    const double *
    RectRegion::coords2D() const {
      return coords_2D_;
    }

    const double *
    RectRegion::coords3D() const {
      return coords_3D_;
    }

    bool
    RectRegion::valid() const {
      return (coords_[0] < coords_[2] && coords_[1] < coords_[3]);
    }

    bool
    RectRegion::intersects(const RectRegion &region) const {
      return (valid() && region.valid() &&
              ((x1() <= region.x2() && x1() >= region.x1()) ||
               (x2() <= region.x2() && x2() >= region.x1())) &&
              ((y1() <= region.y2() && y1() >= region.y1()) ||
               (y2() <= region.y2() && y2() >= region.y1())));
    }


    bool
    RectRegion::inside(double x, double y) const {
      return x >= x1() && x <= x2() && y >= y1() && y <= y2();
    }


    RectRegion
    RectRegion::operator+(const RectRegion &rhs) const 
    {
      if (!valid())
        if (rhs.valid())
          return rhs; // !this->valid() && rhs.valid()
        else
          return RectRegion(); // !this->valid && !rhs.valid()
      else if (!rhs.valid())
        return *this; // this->valid && !rhs.valid()
      
      // this->valid && rhs.valid()
      return RectRegion(Min(x1(), rhs.x1()), 
                        Min(y1(), rhs.y1()),
                        Max(x2(), rhs.x2()),
                        Max(y2(), rhs.y2()));
    }

    RectRegion
    RectRegion::operator-(const RectRegion &rhs) const 
    {
      if (!valid() || !rhs.valid())
        return *this;

      if (!intersects(rhs))
        return RectRegion();

      // this->valid && rhs.valid()
      return RectRegion(Max(x1(), rhs.x1()), 
                        Max(y1(), rhs.y1()),
                        Min(x2(), rhs.x2()),
                        Min(y2(), rhs.y2()));
    }
  }
}

