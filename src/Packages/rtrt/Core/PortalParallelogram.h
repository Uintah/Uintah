
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

 public:

  PortalParallelogram(const Point &p, const Vector &u, const Vector &v)
    : Parallelogram(&portal_m_, p, u, v) { portal_m_.set(p,u,v); }
  virtual ~PortalParallelogram() {}

  PortalMaterial *get_portal() { return &portal_m_; }

  static void attach(PortalParallelogram *a, PortalParallelogram *b)
  {
    PortalMaterial *a_ = a->get_portal();
    PortalMaterial *b_ = b->get_portal();

    a_->attach(b_); /* this call also does b_->attach(a_) */
  }
};

} // end namespace

#endif
