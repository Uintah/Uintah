#ifndef UINTAH_MISCMATH_H
#define UINTAH_MISCMATH_H

namespace Uintah {
  //__________________________________
  //   compute Nan
  double getNan(){
    double nanvalue;
    unsigned int* ntmp = reinterpret_cast<unsigned int*>(&nanvalue);
    ntmp[0] = 0xffff5a5a;
    ntmp[1] = 0xffff5a5a;
    return nanvalue;
  }
  
  double nanValue = getNan();
} // namespace Uintah
#endif
