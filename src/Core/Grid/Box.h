/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#ifndef UINTAH_GRID_BOX_H
#define UINTAH_GRID_BOX_H

#include <Core/Geometry/Point.h>
#include <deque>
#include   <iosfwd>

namespace Uintah {

using std::deque;
using namespace SCIRun;

/**************************************

  CLASS
        Box
   
  GENERAL INFORMATION

        Box.h

        Steven G. Parker
        Department of Computer Science
        University of Utah

        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
        
        Copyright (C) 2000 SCI Group

  KEYWORDS
        Box

  DESCRIPTION
        Long description...
  
  WARNING
  
****************************************/

   class Box {
   public:
      inline Box() {}
      inline ~Box() {}

      inline Box(const Box& copy)
         : d_lower(copy.d_lower), d_upper(copy.d_upper)
      {
      }

      inline Box(const Point& lower, const Point& upper)
         : d_lower(lower), d_upper(upper)
      {
      }

      inline Box& operator=(const Box& copy)
      {
         d_lower = copy.d_lower;
         d_upper = copy.d_upper;
         return *this;
      }

      // A 'Box' is specified by two (3D) points.  However, all components (x,y,z) of the fist point (p1) must be less
      // than the corresponding component of the 2nd point (p2), or the Box is considered 'degenerate()'.  If you know
      // that your box is valid, this function will run through each component and make sure that the lesser value is
      // stored in p1 and the greater value in p2.
      void fixBoundingBox();

       bool overlaps(const Box&, double epsilon=1.e-6) const;
      
      
      // Do not use this for determining if a point is within a patch
      // The logic is incorrect if there are multiple patches
       bool contains(const Point& p) const {
         return p.x() >= d_lower.x() && p.y() >= d_lower.y()
            && p.z() >= d_lower.z() && p.x() < d_upper.x()
            && p.y() < d_upper.y() && p.z() < d_upper.z();
      }

      inline Point lower() const {
         return d_lower;
      }
      inline Point upper() const {
         return d_upper;
      }
      inline Box intersect(const Box& b) const {
         return Box(Max(d_lower, b.d_lower),
                    Min(d_upper, b.d_upper));
      }
      inline bool degenerate() const {
         return d_lower.x() >= d_upper.x() || d_lower.y() >= d_upper.y() || d_lower.z() >= d_upper.z();
      }

       static deque<Box> difference(const Box& b1, const Box& b2);
       static deque<Box> difference(deque<Box>& boxSet1, deque<Box>& boxSet2);

       friend std::ostream& operator<<(std::ostream& out, const Uintah::Box& b);

   private:
      Point d_lower;
      Point d_upper;
   };

} // End namespace Uintah

#endif
