#include <Packages/Remote/Tools/Math/Perlin.h>

#include <Packages/Remote/Tools/Util/Utils.h>

#include <math.h>

namespace Remote {
int Perlin::Pr[4][MAX_OCTAVES];
double Perlin::Ampl[MAX_OCTAVES];
bool Perlin::RandFilled = false;

// XXX
int Perlin::Histogram[100];

bool Perlin::IsPrime(int n)
{
  if(!(n & 1))
    return false;

  for(int i=3; i*i <= n; i+=2)
      if(!(n % i))
	  return false;

  return true;
}

int Perlin::FindRandPrime(int i0, int i1)
{
  while(1)
    {
      // Get an odd random number in the range.
      int i = (LRand() % ((i1 - i0) / 2)) * 2 + 1;
      i = (i0 & (~1)) + i; 

      // Find the next largest prime.
      for( ; i < i1; i+= 2)
	  if(IsPrime(i))
	      return i;
    }
}

void Perlin::ComputeMax()
{
  MaxVal = 0;
  for(int i = 0; i < NOct; i++)
    {
      Ampl[i] = pow(Persist, double(i));
      MaxVal += Ampl[i];
    }
}

void Perlin::SetNumOctaves(int Oct)
{
  NOct = Oct;
  ComputeMax();
}

void Perlin::SetPersistance(double persist)
{
  Persist = persist;
  ComputeMax();
}

void Perlin::FillRandSeeds()
{
  if(RandFilled)
    return;

  RandFilled = true;

  SRand();
  for(int i=0; i<MAX_OCTAVES; i++)
    {
      Pr[0][i] = FindRandPrime(200, 1000);
      Pr[1][i] = FindRandPrime(10000, 20000);
      Pr[2][i] = FindRandPrime(500000, 1000000);
      Pr[3][i] = FindRandPrime(1000000000, 2000000000);
    }
}

// The first time around, fill in the random number tables.
Perlin::Perlin()
{
  memset(Histogram, 0, sizeof(Histogram));

  Persist = 0.5;
  NOct = 1;
  ComputeMax();

  FillRandSeeds();
}

Perlin::Perlin(double persist, int Oct)
{
  Persist = persist;
  NOct = Oct;
  ComputeMax();

  FillRandSeeds();
}

// X and Y should be in 0.0 -> pretty big.
double Perlin::InterpolatedNoise(double x, int Oct)
{
  int xi = int(floor(x));
  double xf = x - xi;

  double v0 = RandNoise(xi, Oct);
  double v1 = RandNoise(xi+1, Oct);

  double v = Interp(v0, v1, xf);

  return v;
}

// X and Y should be in 0.0 -> pretty big.
double Perlin::InterpolatedNoise(double x, double y, int Oct)
{
  int xi = int(floor(x));
  int yi = int(floor(y));
  double xf = x - xi;
  double yf = y - yi;

  double v00 = RandNoise(xi, yi, Oct);
  double v01 = RandNoise(xi, yi+1, Oct);
  double v10 = RandNoise(xi+1, yi, Oct);
  double v11 = RandNoise(xi+1, yi+1, Oct);

  double v0 = Interp(v00, v01, yf);
  double v1 = Interp(v10, v11, yf);
  double v = Interp(v0, v1, xf);

  return v;
}

// X and Y should be in 0.0 -> pretty big.
double Perlin::InterpolatedNoise(double x, double y, double z, int Oct)
{
#if 1
  double xf = x+14639.0;
  double yf = y+6547.0;
  double zf = z+7123.0;
  int xi = int(xf);
  int yi = int(yf);
  int zi = int(zf);
  xf = xf - xi;
  yf = yf - yi;
  zf = zf - zi;
#else
  int xi = int(floor(x));
  int yi = int(floor(y));
  int zi = int(floor(z));
  double xf = x - xi;
  double yf = y - yi;
  double zf = z - zi;
#endif

  double v000 = RandNoise(xi, yi, zi, Oct);
  double v001 = RandNoise(xi, yi, zi+1, Oct);
  double v010 = RandNoise(xi, yi+1, zi, Oct);
  double v011 = RandNoise(xi, yi+1, zi+1, Oct);
  double v100 = RandNoise(xi+1, yi, zi, Oct);
  double v101 = RandNoise(xi+1, yi, zi+1, Oct);
  double v110 = RandNoise(xi+1, yi+1, zi, Oct);
  double v111 = RandNoise(xi+1, yi+1, zi+1, Oct);

  double v00 = Interp(v000, v001, zf);
  double v01 = Interp(v010, v011, zf);
  double v10 = Interp(v100, v101, zf);
  double v11 = Interp(v110, v111, zf);

  double v0 = Interp(v00, v01, yf);
  double v1 = Interp(v10, v11, yf);

  double v = Interp(v0, v1, xf);

  return v;
}

// X and Y should be on 0.0 -> 1.0.
double Perlin::Noise(double x)
{
  double sum = 0.0;
  
  for(int i = 0; i < NOct; i++)
    {
      double freq = 1 << i;
      sum += InterpolatedNoise(x * freq, i) * Ampl[i];
    }

  return sum;
}

// X and Y should be on 0.0 -> 1.0.
double Perlin::Noise(double x, double y)
{
  double sum = 0.0;
  
  for(int i = 0; i < NOct; i++)
    {
      double freq = 1 << i;
      sum += InterpolatedNoise(x * freq, y * freq, i) * Ampl[i];
    }

  return sum;
}

// X and Y should be on 0.0 -> 1.0.
double Perlin::Noise(double x, double y, double z)
{
  double sum = 0.0;
  
  for(int i = 0; i < NOct; i++)
    {
      double freq = 1 << i;
      sum += InterpolatedNoise(x * freq, y * freq, z * freq, i) * Ampl[i];
    }

  // BuckIt(sum);
  return sum;
}
} // End namespace Remote


