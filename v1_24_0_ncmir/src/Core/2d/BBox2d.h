/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  BBox2d.h: 
 *
 *  Written by:
 *   Author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef BBox2d_h
#define BBox2d_h 


#include <Core/2d/Point2d.h>

namespace SCIRun {

class Piostream;

class BBox2d {

protected:
  friend void Pio( Piostream &, BBox2d& );
  
  bool have_some;
  Point2d min_, max_;

public:
  BBox2d();
  ~BBox2d();
  BBox2d(const BBox2d&);
  BBox2d(const Point2d& min, const Point2d& max);
  inline int valid() const {return have_some;}
  void reset();
  void extend(const Point2d& p);
  void extend(const BBox2d& b);

  Point2d min() const;
  Point2d max() const;

  int inside(const Point2d &p) const {
    return have_some 
      && p.x()>=min_.x() && p.y()>=min_.y() 
      && p.x()<=max_.x() && p.y()<=max_.y();
  }

};

} // End namespace SCIRun


#endif
