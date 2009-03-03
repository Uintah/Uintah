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



#ifndef CUTVOLUMEDPY_H
#define CUTVOLUMEDPY_H 1

#include <X11/Xlib.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/ColorMap.h>

/*
This is a new VolumeDpy used for CutPlanes.
The histogram is colored according to a ColorMap.
The Dpy can modify and save the associated ColorMap.
*/

namespace rtrt {

using SCIRun::Runnable;

class CutVolumeDpy : public VolumeDpy {
 protected:
  virtual void draw_hist(bool redraw_hist);

  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);
  virtual void key_pressed(unsigned long key);

  void insertcolor(int x);
  void selectcolor(int x);
  void movecolor(int x);
  int selcolor;
 public:
  CutVolumeDpy(float isoval=-123456, ColorMap *cmap=0);

  ColorMap *cmap;
};

} // end namespace rtrt


#endif



