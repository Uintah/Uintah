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



#ifndef CUTPLANEDPY_H
#define CUTPLANEDPY_H 1

#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Thread/Runnable.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/Ball.h>
#include <X11/Xlib.h>


/*
New Plane Display.
Updated for Cutting Planes to include on/off, and track ball rotation.
Also, this attempts to keep track of a center point around which to rotate.
*/

namespace rtrt {

using SCIRun::Transform;
using SCIRun::Vector;
using SCIRun::Point;

class CutPlaneDpy : public PlaneDpy {
 protected:
    virtual void move(int x, int y);
    BallData *ball;
    bool rotsphere;
    Transform *prev_trans;

    virtual void init();
    virtual void display();
    virtual void resize(const int width, const int height);
    virtual void key_pressed(unsigned long key);
    virtual void button_pressed(MouseButton button, const int x, const int y);
    virtual void button_released(MouseButton button, const int x, const int y);
    virtual void button_motion(MouseButton button, const int x, const int y);

    bool prev_doanimate;
 public:
    Point cen; //used for rotation with middle mouse button
    bool on; //turns on and off the associated cutgroup
    double dscale; //changes rate at which D bar moves, larger models need > [-1..1]

    CutPlaneDpy(const Vector& v, const Point& p);
    CutPlaneDpy(const Vector& v, const double d);
    virtual ~CutPlaneDpy();
    void redisplay() { redraw = true; };

    bool doanimate;
    void Toggle_doanimate() { doanimate = !doanimate; }
    void animate(double t);
};

} // end namespace rtrt

#endif

