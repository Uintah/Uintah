#ifndef _arcball_h
#define _arcball_h

#include <Packages/Remote/Tools/Math/Vector.h>
#include <Packages/Remote/Tools/Math/Matrix44.h>

namespace Remote {
using namespace Remote::Tools;

struct ArcBall
{
  Vector Axis;
  double angle;
  Matrix44 M;
  Vector Old; // Last point on the sphere.

  // Takes an X and Y on -1 to 1.
  void Touch(double x, double y)
  {
    double rSqr = Sqr(x)+Sqr(y);
    if(rSqr > 1)
      return;
    
    double z = sqrt(1.0 - rSqr);

    Old = Vector(x, y, z);
    Old.normalize();
  }

  // Takes an X and Y on -1 to 1.
  void Move(double x, double y)
  {
    double rSqr = Sqr(x)+Sqr(y);
    if (rSqr > 1)
      return;
    
    double z = sqrt(1.0 - rSqr);

    Vector P = Vector(x, y, z); // Point on the sphere.

    Vector Ax = Cross(Old, P);
    Ax.normalize();
    double Ang = acos(Dot(Old, P));

    Old = P;

    Matrix44 R;
    R.Rotate(Ang, Ax);

    M = R * M;
  }
};
} // End namespace Remote


#endif

