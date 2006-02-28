#ifndef UINTAH_HOMEBREW_PARTIAL_ELLIPTIC_CRACK_H
#define UINTAH_HOMEBREW_PARTIAL_ELLIPTIC_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>


namespace Uintah {

class PartialEllipticCrack : public CrackGeometry
{
  public:
     // Constructors
     PartialEllipticCrack(ProblemSpecP& ps);
     PartialEllipticCrack(const PartialEllipticCrack& copy);

     // Destructor
     virtual ~PartialEllipticCrack();
     virtual void readCrack(ProblemSpecP&);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);

 private:
     int NCells, CrkFrtSegID;
     double Extent;
};

}//End namespace Uintah

#endif  /* __PARTIAL_ELLIPTIC_CRACK_H__*/
