#ifndef _FRACTUREFACTORY_H_
#define _FRACTUREFACTORY_H_

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {
class Fracture;
    
class FractureFactory {
public:
	
  static Fracture* create(const ProblemSpecP& ps);
  
};
} // End namespace Uintah
    


#endif /* _FRACTUREFACTORY_H_ */

