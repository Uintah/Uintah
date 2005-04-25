
#ifndef PARALLELOGRAM2_H
#define PARALLELOGRAM2_H 1

#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/MultiMaterial.h>

//
// THIS CLASS INTENDED FOR SIGGRAPH02 DEMO ONLY
//


namespace rtrt {

class Parallelogram2 : public Parallelogram
{
  
 public:
  
  Parallelogram2(Material *m, const Point &p, const Vector &v0, 
                 const Vector &v1)
    : Parallelogram(m,p,v0,v1) {}
  virtual ~Parallelogram2() {}

  virtual void animate(double t, bool& changed);
  
};

} // end namespace

#endif
