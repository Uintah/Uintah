#ifndef UINTAH_HOMEBREW_NULL_CRACK_H
#define UINTAH_HOMEBREW_NULL_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>


namespace Uintah {

class NullCrack : public CrackGeometry
{
  public:
     // Constructors
     NullCrack(ProblemSpecP& ps);
     NullCrack(const NullCrack& copy);

     // Destructor
     virtual ~NullCrack();
     virtual void readCrack(ProblemSpecP&);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);
};

}//End namespace Uintah

#endif  /* __NULL_CRACK_H__*/
