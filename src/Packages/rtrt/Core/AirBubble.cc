
/****************************************************************\
 *                                                              *
 * filename : AirBubble.cc                                      *
 * author   : R. Keith Morley                                   *
 * last mod : 07/25/02                                          *
 *                                                              *
\****************************************************************/

#include <Packages/rtrt/Core/AirBubble.h>
#include <Packages/rtrt/Core/BBox.h>

using namespace rtrt;

AirBubble::AirBubble(Material* matl, const Point& cen, double radius, double vert, double horiz, double _speed)
   : Sphere(matl, cen, radius), ocen(cen), vertical(vert / speed), horizontal(horiz), speed(_speed)
{}

AirBubble::~AirBubble()
{}

void AirBubble::animate(double t, bool& changed)
{
   changed = true;

   // figure out the height of the bubble
   int t_int = (int)(t);
   double height = (int)(t_int / vertical);
   height =  speed * (t - height * vertical);

   // now calculate the forizontal offset from vertical path  
   cen = ocen + height * Vector(0, 0, 1);
   Vector horiz = horizontal * noise.dnoise(4.0 * Vector(cen)); 
   cen = Point(cen.x() + horiz.x(), cen.y() + horiz.y(), cen.z());
}

void AirBubble::compute_bounds(BBox& bbox, double offset)
{
   Vector motion(0, 0, ocen.z() + vertical);
   bbox.extend(ocen        , radius + horizontal +offset);
   bbox.extend(ocen +motion, radius + horizontal +offset);
}

