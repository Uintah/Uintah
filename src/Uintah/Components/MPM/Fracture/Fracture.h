#ifndef __FRACTURE_H__
#define __FRACTURE_H__

namespace Uintah {
namespace MPM {

class Fracture {
public:
  static void   fractureParametersInitialize();
  static void   materialDefectsInitialize();
  
private:
  static double   d_crackDensity;
  static double   d_averageMicrocrackLength;
  static double   d_materialToughness;
};

} //namespace MPM
} //namespace Uintah

#endif //__FRACTURE_H__
