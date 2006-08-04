#ifndef UINTAH_HOMEBREW_ARC_CRACK_H
#define UINTAH_HOMEBREW_ARC_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>


namespace Uintah {

class ArcCrack : public CrackGeometry
{
  public:
     // Constructors
     ArcCrack(ProblemSpecP& ps);
     ArcCrack(const ArcCrack& copy);

     // Destructor
     virtual ~ArcCrack();
     virtual void readCrack(ProblemSpecP&);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);

 private:
     int NCells, CrkFrtSegID;

};

}//End namespace Uintah

#endif  /* __ARC_CRACK_H__*/
