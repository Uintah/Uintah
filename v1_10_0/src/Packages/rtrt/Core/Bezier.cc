#include <Packages/rtrt/Core/Bezier.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* bezier_maker() {
  return new Bezier();
}

// initialize the static member type_id
PersistentTypeID Bezier::type_id("Bezier", "Object", bezier_maker);

Bezier::Bezier(Material *mat, Mesh *m, 
	       double u0, double u1,
	       double v0, double v1) : Object(mat)
{
  control = m;
  local = m->Copy();
  isleaf = 1;
  ustart = u0;
  ufinish = u1;
  vstart = v0;
  vfinish = v1;
  uguess = .5*(ustart+ufinish);
  vguess = .5*(vstart+vfinish);
  bbox = 0;
}

Bezier::Bezier(Material *mat, Mesh *g, 
	       Mesh *l, 
	       double u0, double u1,
	       double v0, double v1) : Object(mat)
{
  control = g;
  local = l;
  isleaf = 1;
  ustart = u0;
  vstart = v0;
  ufinish = u1;
  vfinish = v1;
  uguess = .5*(ustart+ufinish);
  vguess = .5*(vstart+vfinish);
  bbox = 0;
}

void Bezier::preprocess(double, int&, int& scratchsize)
{
  scratchsize=Max(scratchsize, control->get_scratchsize());
}

const int BEZIER_VERSION = 1;

void 
Bezier::io(SCIRun::Piostream &str)
{
  str.begin_class("Bezier", BEZIER_VERSION);
  Object::io(str);
  SCIRun::Pio(str, control);
  SCIRun::Pio(str, local);
  SCIRun::Pio(str, ne);
  SCIRun::Pio(str, nw);
  SCIRun::Pio(str, se);
  SCIRun::Pio(str, sw);
  SCIRun::Pio(str, bbox);
  SCIRun::Pio(str, isleaf);
  SCIRun::Pio(str, ustart);
  SCIRun::Pio(str, ufinish);
  SCIRun::Pio(str, vstart);
  SCIRun::Pio(str, vfinish);
  SCIRun::Pio(str, uguess);
  SCIRun::Pio(str, vguess);
 str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Bezier*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Bezier::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Bezier*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
