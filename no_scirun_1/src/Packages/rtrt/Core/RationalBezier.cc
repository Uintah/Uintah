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


void RationalBezier::preprocess(double, int&, int& scratchsize)
{
  scratchsize=Max(scratchsize, control->get_scratchsize());
}

