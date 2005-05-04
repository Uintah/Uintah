#ifndef _perlin_h
#define _perlin_h

#include <Packages/Remote/Tools/Math/Vector.h>

#define MAX_OCTAVES 16

namespace Remote {
class Perlin
{
  int NOct;
  double Persist;
  static bool RandFilled;
  static int Pr[4][MAX_OCTAVES];
  static double Ampl[MAX_OCTAVES];
  static int Histogram[100]; // XXX

  bool IsPrime(int n);
  int FindRandPrime(int i0, int i1);
  void ComputeMax();
  void FillRandSeeds();

  // XXX Test random using histogram
  inline void BuckIt(double t)
  {
    t *= 80.0;
    t += 10.0;

    if(t >=0 && t <= 99)
      Histogram[int(t)]++;
  }

  // These ones return a random double on -1.0 -> 1.0 based on the seed x,y,z.
  inline double RandNoise(int x, int i)
  {
    x = (x << 13) ^ x;
    // -1 to 1
    // return 1.0 - double((x * (x*x* Pr[1][i] + Pr[2][i]) + Pr[3][i]) & 0x7fffffff) / 1073741823.0;
    // 0 to 1
    return double((x * (x*x* Pr[1][i] + Pr[2][i]) + Pr[3][i]) & 0x7fffffff) / 2147483647.0;
  }

  inline double RandNoise(int x, int y, int i)
  {
    return RandNoise(y * Pr[0][i] + x, i);
  }

  inline double RandNoise(int x, int y, int z, int i)
  {
    double t = RandNoise(z * Pr[1][i] + y * Pr[0][i] + x, i);
    // printf("%d %d %d %f\n", x, y, z, t); // XXX
    // BuckIt(t);
    return t;
  }

  // Smoothly interpolates between a and b.
  inline double Interp(double a, double b, double t)
  {
    double f = t*t*(3-2*t);
    
    return a + (b - a) * f;
  }

public:
  Perlin();
  Perlin(double persist, int Oct);

#if 0
  inline ~Perlin() // XXX
  {
    for(int i=0; i < 100; i++)
      printf("%f %d\n", (double(i)-10.0)/80.0, Histogram[i]);
  }
#endif

  double MaxVal; // Tells the maximum value ofthe noise function. Min = -Max.

  // X Y and Z should be on 0.0 -> 1.0.
  double Noise(double x);
  double Noise(double x, double y);
  double Noise(double x, double y, double z);
  inline double Noise(const Vector &V) {return Noise(V.x, V.y, V.z);}

  // This should really be private, but sometimes I want to call it directly.
  // X and Y should be in 0.0 -> pretty big.
  double InterpolatedNoise(double x, int Oct);
  double InterpolatedNoise(double x, double y, int Oct);
  double InterpolatedNoise(double x, double y, double z, int Oct);

  void SetNumOctaves(int Oct); // Oct >= 1.
  void SetPersistance(double persist); // 0 < persist < 1. 0.5 is good.
};

} // End namespace Remote


#endif
