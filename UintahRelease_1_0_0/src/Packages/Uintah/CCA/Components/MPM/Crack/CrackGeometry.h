#ifndef UINTAH_HOMEBREW_CRACK_GEOMETRY_H
#define UINTAH_HOMEBREW_CRACK_GEOMETRY_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <vector>

namespace Uintah {
  using SCIRun::Point;
  using SCIRun::IntVector;

class CrackGeometry
{
  public:
     // Constructors
     CrackGeometry();

     // Destructor
     virtual ~CrackGeometry();
     virtual void readCrack(ProblemSpecP&) = 0;
     virtual void outputInitialCrackPlane(int i) = 0;
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes) = 0;

     bool twoLinesCoincide(Point& p1, Point& p2, Point& p3, Point& p4);

 protected:
     vector<Point> vertices;
};

}//End namespace Uintah

#endif  /* __CRACK_GEOMETRY_H__*/
