#ifndef UINTAH_HOMEBREW_CURVED_QUAD_CRACK_H
#define UINTAH_HOMEBREW_CURVED_QUAD_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>


namespace Uintah {

class CurvedQuadCrack : public CrackGeometry
{
  public:
     // Constructors
     CurvedQuadCrack(ProblemSpecP& ps);
     CurvedQuadCrack(const CurvedQuadCrack& copy);

     // Destructor
     virtual ~CurvedQuadCrack();
     virtual void readCrack(ProblemSpecP& ps);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);

 private:

     int NStraightSides;
     vector<Point> PtsSide2,PtsSide4;
     vector<bool> AtFront;
     int Repetition;
     Vector Offset;
};


}//End namespace Uintah

#endif  /* __CURVED_QUAD_CRACK_H__*/
