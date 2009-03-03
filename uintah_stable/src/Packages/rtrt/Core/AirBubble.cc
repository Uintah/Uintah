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
   : Sphere(matl, cen, radius), ocen(cen), speed(_speed),
     vertical(vert / speed), horizontal(horiz)
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

