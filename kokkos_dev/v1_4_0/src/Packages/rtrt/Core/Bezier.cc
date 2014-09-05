#include <Packages/rtrt/Core/Bezier.h>

using namespace rtrt;

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

