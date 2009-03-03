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



#ifndef TEXTUREDTRI2_H
#define TEXTUREDTRI2_H 1

#include <Packages/rtrt/Core/TexturedTri.h>

#define TEXSCROLLSPEED 1

namespace rtrt {

class TexturedTri2 : public TexturedTri
{

 protected:

  Transform tex_trans_;

 public:

  TexturedTri2(Material *m, const Point &p0, const Point &p1, const Point &p2)
    : TexturedTri(m,p0,p1,p2) { tex_trans_.load_identity(); }
  virtual ~TexturedTri2() {}

  void translate_tex(const Vector &v)
    { tex_trans_.pre_translate(v); }

  void scale_tex(double x, double y, double z)
    { tex_trans_.pre_scale( Vector(x,y,z) ); }
    
  void uv(UV& uv, const Point&, const HitInfo& hit)
  {
    Point tp = t1+((ntu*((double*)hit.scratchpad)[1])+
                   (ntv*((double*)hit.scratchpad)[0]));

    Point xtp = tex_trans_.project(tp);
    
    uv.set(xtp.x(),xtp.y());
  }

  virtual void animate(double t, bool& changed)
  {
    translate_tex(Vector(0,t*TEXSCROLLSPEED*.0001,0));
    changed=true;
  }
};

}

#endif
