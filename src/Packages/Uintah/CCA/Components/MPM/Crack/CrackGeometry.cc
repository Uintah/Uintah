#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>
#include <Core/Geometry/Vector.h>
#include <math.h>

using namespace Uintah;
using namespace SCIRun;


CrackGeometry::CrackGeometry()
{
}


CrackGeometry::~CrackGeometry()
{
  // Destructor
  // Do nothing
}

bool CrackGeometry::twoLinesCoincide(Point& p1, Point& p2, Point& p3, 
                                     Point& p4)
{
  // Check for coincidence between line segment p3-p4 and line
  // segment p1 - p2
  double l12 = (p2.asVector() - p1.asVector()).length();
  double l31 = (p3.asVector() - p1.asVector()).length();
  double l32 = (p3.asVector() - p2.asVector()).length();
  double l41 = (p4.asVector() - p1.asVector()).length();
  double l42 = (p4.asVector() - p2.asVector()).length();

  if (fabs(l31+l32-l12)/l12 < 1.e-6 && fabs(l41+l42-l12)/l12 < 1.e-6 && l41>l31)
    return true;
  else
    return false;


}
