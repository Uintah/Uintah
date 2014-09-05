#include <Packages/rtrt/Core/RationalBezier.h>

using namespace rtrt;

RationalBezier::RationalBezier(Material *mat, RationalMesh *m, 
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
  bbox = new BBox();
  control->compute_bounds(*bbox);
}

RationalBezier::RationalBezier(Material *mat, RationalMesh *g, 
			       RationalMesh *l, 
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

void RationalBezier::Out(int i=0, int j=0)
{
  if (isleaf) {
    control->Out(i,j);
  }
  else {
    i*=4;
    j*=4;
    nw->Out(i,j);
    ne->Out(i,j+1);
    se->Out(i+1,j+1);
    sw->Out(i+1,j);
  }
}

inline void RationalBezier::Print()
{
  if (isleaf) {
    local->Print();
    printf("Ustart:%lf Ustop:%lf Vstart:%lf Vstop:%lf\n",
	   ustart,ufinish,vstart,vfinish);
  }
  else {
    nw->Print();
    ne->Print();
    se->Print();
    sw->Print();
  }
}

void RationalBezier::preprocess(double, int&, int& scratchsize)
{
  scratchsize=Max(scratchsize, control->get_scratchsize());
}

