#ifndef UINTAH_HOMEBREW_QUAD_CRACK_H
#define UINTAH_HOMEBREW_QUAD_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>

namespace Uintah {

class QuadCrack : public CrackGeometry
{
  public:
     // Constructors
     QuadCrack(ProblemSpecP& ps);
     QuadCrack(const QuadCrack& copy);

     // Destructor
     virtual ~QuadCrack();
     virtual void readCrack(ProblemSpecP&);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);

 protected:
     void GetGlobalCoordinates(const int &, const double &, 
                               const double&, Point&);
 private:
     int N12, N23;
     vector<bool> AtFront;
     int Repetition;
     Vector Offset;
};


}//End namespace Uintah

#endif  /* __QUAD_CRACK_H__*/
