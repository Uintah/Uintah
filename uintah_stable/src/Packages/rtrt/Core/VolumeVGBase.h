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



#ifndef RTRT_VolumeVGBASE_H
#define RTRT_VolumeVGBASE_H 1

#include <Packages/rtrt/Core/Object.h>

namespace rtrt {
  
class Hist2DDpy;

class VolumeVGBase : public Object {
protected:
  Hist2DDpy* dpy;
public:
  VolumeVGBase(Material* matl, Hist2DDpy* dpy);
  virtual ~VolumeVGBase();
  virtual void animate(double t, bool& changed);
  virtual void compute_hist(int nxhist, int nyhist, int** hist,
			    float vdatamin, float vdatamax,
			    float gdatamin, float gdatamax)=0;
  virtual void get_minmax(float& vmin, float& vmax,
			  float& gmin, float& gmax)=0;
};

} // end namespace rtrt

#endif // #ifndef RTRT_VolumeVGBASE_H
