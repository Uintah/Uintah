#ifndef __ThermalContactFactory__
#define __ThermalContactFactory__

#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
   namespace MPM {
      class ThermalContact;

class ThermalContactFactory {
public:
  static ThermalContact* create(ProblemSpecP& ps);

};
      
} // end namespace MPM
} // end namespace Uintah

#endif //__ThermalContactFactory__

// $Log$
// Revision 1.1  2000/06/20 03:20:23  tan
// Added ThermalContactFactory class to interface with ProblemSpecification.
//
