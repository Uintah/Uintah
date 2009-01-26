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



#ifndef PLANEDPY_H
#define PLANEDPY_H 1

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <X11/Xlib.h>

namespace rtrt {

using SCIRun::Runnable;
using SCIRun::Vector;
using SCIRun::Point;

class PlaneDpy : public DpyBase {
protected:
  int starty, startx;
  virtual void move(int x, int y);
  virtual void offset(int x, int y);
  
  virtual void init();
  virtual void display();
  virtual void resize(const int width, const int height);
  virtual void key_pressed(unsigned long key);
  virtual void button_pressed(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);
  
public:
  Vector n;
  double d;
  PlaneDpy(const Vector& v, const Point& p,
	   bool active = true, bool use_material = true);
  PlaneDpy(const Vector& v, const double d,
	   bool active = true, bool use_material = true);
  virtual ~PlaneDpy();

  bool active;
  bool use_material;
};

} // end namespace rtrt

#endif

