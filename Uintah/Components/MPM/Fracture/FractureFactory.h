#ifndef _FRACTUREFACTORY_H_
#define _FRACTUREFACTORY_H_

#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/SimulationStateP.h>

namespace Uintah {
namespace MPM {

class Fracture;
    
class FractureFactory {
public:
	
  static Fracture* create(const ProblemSpecP& ps,SimulationStateP& ss);
  
};
    
} // end namespace MPM
} // end namespace Uintah


#endif /* _FRACTUREFACTORY_H_ */

// $Log$
// Revision 1.1  2000/05/10 05:07:53  tan
// Basic structure of FractureFactory class.
//
