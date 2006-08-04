#ifndef UINTAH_HOMEBREW_ELLIPTIC_CRACK_H
#define UINTAH_HOMEBREW_ELLIPTIC_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>


namespace Uintah {

class EllipticCrack : public CrackGeometry
{
  public:
     // Constructors
     EllipticCrack(ProblemSpecP& ps);
     EllipticCrack(const EllipticCrack& copy);

     // Destructor
     virtual ~EllipticCrack();
     virtual void readCrack(ProblemSpecP&);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);

 private:
     int NCells, CrkFrtSegID;
};

}//End namespace Uintah

#endif  /* __ELLIPTIC_CRACK_H__*/
