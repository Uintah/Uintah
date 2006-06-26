
#ifndef BV1_H
#define BV1_H 1

#include <Packages/rtrt/Core/Object.h>

namespace rtrt {
class BV1;
}
namespace SCIRun {
void Pio(Piostream&, rtrt::BV1*&);
}

namespace rtrt {
struct BV1Tree;

class BV1 : public Object {
  Object* obj;

  BV1Tree* normal_tree;
  BV1Tree* light_tree;

  BV1Tree* make_tree(double maxradius);
  void make_tree(int nprims, Object** prims, double* slabs, int which=0);
  void finishit(double* slabs, Array1<Object*>& prims, int primStart);
public:
  BV1(Object* obj);
  BV1() : Object(0) {} // empty default constructor for Pio.
  virtual ~BV1();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, BV1*&);

    virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual void multi_light_intersect(Light* light, const Point& orig,
				     const Array1<Vector>& dirs,
				     const Array1<Color>& attens,
				     double dist,
				     DepthStats* st, PerProcessorContext* ppc);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void collect_prims(Array1<Object*>& prims);
};

} // end namespace rtrt

#endif
