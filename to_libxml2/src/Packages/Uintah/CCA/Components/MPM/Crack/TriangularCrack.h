#ifndef UINTAH_HOMEBREW_TRIANGULAR_CRACK_H
#define UINTAH_HOMEBREW_TRIANGULAR_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>


namespace Uintah {

class TriangularCrack : public CrackGeometry
{
  public:
     // Constructors
     TriangularCrack(ProblemSpecP& ps);
     TriangularCrack(const TriangularCrack& copy);

     // Destructor
     virtual ~TriangularCrack();
     virtual void readCrack(ProblemSpecP&);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);

 private:
     int NCells;
     vector<bool> AtFront;
     int Repetition;
     Vector Offset;
};


}//End namespace Uintah

#endif  /* __TRIANGULAR_CRACK_H__*/
