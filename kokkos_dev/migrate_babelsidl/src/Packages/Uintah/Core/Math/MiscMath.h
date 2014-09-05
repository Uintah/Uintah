#ifndef UINTAH_MISCMATH_H
#define UINTAH_MISCMATH_H

namespace Uintah {
  //__________________________________
  //   compute Nan
  inline double getNan(){
    double nanvalue;
    unsigned int* ntmp = reinterpret_cast<unsigned int*>(&nanvalue);
    ntmp[0] = 0xffff5a5a;
    ntmp[1] = 0xffff5a5a;
    return nanvalue;
  }
} // namespace Uintah
#endif
