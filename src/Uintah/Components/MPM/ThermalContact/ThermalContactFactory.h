#ifndef __ThermalContactFactory__
#define __ThermalContactFactory__

#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/SimulationStateP.h>

namespace Uintah {
   namespace MPM {
      class ThermalContact;

class ThermalContactFactory {
public:
  static ThermalContact* create(const ProblemSpecP& ps,SimulationStateP& d_sS);

};
      
} // end namespace MPM
} // end namespace Uintah

#endif //__ThermalContactFactory__

// $Log$
// Revision 1.2  2000/06/20 04:14:20  tan
// WHen SerialMPM::d_thermalContactModel != NULL, heat conduction will be included
// in MPM algorithm.  The d_thermalContactModel is set by ThermalContactFactory
// according to the information in ProblemSpec from input file.
//
// Revision 1.1  2000/06/20 03:20:23  tan
// Added ThermalContactFactory class to interface with ProblemSpecification.
//
