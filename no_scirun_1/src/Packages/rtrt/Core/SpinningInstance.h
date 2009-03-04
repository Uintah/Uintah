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


#ifndef SPINNINGINSTANCE_H
#define SPINNINGINSTANCE_H

#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/BBox.h>

namespace rtrt {

  class SpinningInstance: public Instance
    {
      Transform *location_trans;
      BBox bbox_orig;
      bool dorotate;
      double ctime;
      CutPlaneDpy *cpdpy;
    public:
      Point cen;
      Vector axis;
      double rate;

      SpinningInstance(InstanceWrapperObject* o, Transform* trans, Point cen, Vector axis, double rate);
      virtual void compute_bounds(BBox& b, double /*offset*/);
      virtual void animate(double t, bool& changed);
      void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		     PerProcessorContext* ppc);

      void toggleDoSpin() { dorotate = !dorotate; if (cpdpy) cpdpy->doanimate=dorotate; };	
      int doSpin() { if (dorotate) return 1; else return 0; };
      void incMagnification();	
      void decMagnification();	
      void upPole();
      void downPole();
      void addCPDpy(CutPlaneDpy *_cpdpy) { cpdpy = _cpdpy; }
    };
}
#endif
