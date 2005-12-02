
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

