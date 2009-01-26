/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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



/*********************************************************************\
 *                                                                   *
 * filename : AirBubble.cc                                           *
 * author   : R. Keith Morley                                        *
 * last mod : 07/25/02                                               *
 *                                                                   *
 * Air bubble rising to surface.  For use in *ray siggraph demo.     *
 * Air buuble starts at initial cen and rises to cen + vert and      *
 * varies a mas of +- horiz.                                         *
 *                                                                   *
\*********************************************************************/


#ifndef _AIR_BUBBLE_H_
#define _AIR_BUBBLE_H_ 1

#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Noise.h>


namespace rtrt {

class AirBubble : public Sphere {
    Point ocen;
    double speed;        // vertical speed
    double vertical;     // max vertical offset from ocen
    double horizontal;   // max horizontal offset from ocen
    Noise noise;
public:
    AirBubble(Material* matl, const Point& cen, double radius, double vert, double horiz, double _speed);
    virtual ~AirBubble();
    virtual void animate(double t, bool& changed);
    virtual void compute_bounds(BBox&, double offset);
};

} // end namespace rtrt

#endif // _AIR_BUBBLE_H_

