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
