
#ifndef Math_FastNoise_h
#define Math_FastNoise_h 1

#include <Packages/rtrt/Core/Noise.h>

namespace rtrt {
  class FastNoise;
}

namespace SCIRun {
  void Pio(Piostream& stream, rtrt::FastNoise &obj);
}

namespace rtrt {

class FastNoise : public Noise {
  inline double smooth(double);
  inline int get_index(int,int,int,int);
  inline double interpolate(int,double,double,double,double);
public:
  FastNoise(int seed=0, int tablesize=4096);
  virtual ~FastNoise();

  friend void SCIRun::Pio(SCIRun::Piostream& stream, rtrt::FastNoise& obj);
  
  double operator()(const Vector&);
  double operator()(double);
  Vector dnoise(const Vector&);
  Vector dnoise(const Vector&, double&);
};

} // end namespace rtrt

#endif
