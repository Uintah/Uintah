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



#include <Packages/rtrt/Core/BouncingSphere.h>
#include <Packages/rtrt/Core/BBox.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;

BouncingSphere::BouncingSphere(Material* matl,
			       const Point& cen, double radius,
			       const Vector& motion)
    : Sphere(matl, cen, radius), ocen(cen), motion(motion)
{
}

BouncingSphere::~BouncingSphere()
{
}

void BouncingSphere::animate(double t, bool& changed)
{
    changed=true;
    int i=(int)t;
    double f=t-i;
    double tt=3*f*f-2*f*f*f;
    if(i%2){
	cen=ocen+motion*tt;
    } else {
	cen=ocen+motion*(1-tt);
    }
}

void BouncingSphere::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(ocen, radius+offset);
    bbox.extend(ocen+motion, radius+offset);
}
