
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
