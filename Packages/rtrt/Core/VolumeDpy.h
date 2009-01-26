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



#ifndef VOLUMEDPY_H
#define VOLUMEDPY_H 1

#include <X11/Xlib.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/DpyBase.h>

namespace rtrt {

using SCIRun::Runnable;

class VolumeBase;

class VolumeDpy : public DpyBase {
 protected:
  Array1<VolumeBase*> vols;
  int* hist;
  int histmax;
  float datamin, datamax;
  
  void compute_hist();
  virtual void draw_hist(bool redraw_hist);

  virtual void init();
  
  // event variable and handlers
  bool need_hist; // default: true
  bool redraw_isoval; // default: false
  virtual void display();
  virtual void resize(const int width, const int height);
  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);

  void move_isoval(const int x);
  float new_isoval;
public:
  float isoval;
  VolumeDpy(float isoval=-123456);
  void attach(VolumeBase*);
  virtual ~VolumeDpy();
  virtual void animate(bool& changed);

  // This will set new_isoval to new_val if it lies within the range
  // [datamin, datamax].
  void change_isoval(float new_val);
  // This changes what datamin and datamax are
  void set_minmax(float min, float max);
};

} // end namespace rtrt


#endif
