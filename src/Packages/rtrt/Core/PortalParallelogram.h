
#ifndef PORTALPARALLELOGRAM_H
#define PORTALPARALLELOGRAM_H 1

#include <Packages/rtrt/Core/PortalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>

namespace rtrt {

class PortalParallelogram : public Parallelogram
{

 protected:

  PortalMaterial      portal_m_;
  PortalParallelogram *other_end_;

  void set_opposite_end(PortalParallelogram *b) { other_end_ = b; set(); }
  PortalParallelogram *get_opposite_end() { return other_end_; }
  void set()
  {
    if (other_end_)
      portal_m_.set(anchor, u, v,
                    other_end_->get_anchor(), 
                    other_end_->get_u(), other_end_->get_v());
  }

 public:

  PortalParallelogram(const Point &p, const Vector &u, const Vector &v,
                      PortalParallelogram *other=0)
    : Parallelogram(&portal_m_, p, u, v), other_end_(other) {}
  virtual ~PortalParallelogram() {}

  virtual void preprocess(double /*maxradius*/, 
                          int& /*pp_offset*/, 
                          int& /*scratchsize*/)
  {
    set();
  }
  
  void attach(PortalParallelogram *a)
  {
    set_opposite_end(a);
    a->set_opposite_end(this);
  }

  static void attach(PortalParallelogram *a, PortalParallelogram *b)
  {
    a->set_opposite_end(b);
    b->set_opposite_end(a);
  }
};

} // end namespace

#endif
