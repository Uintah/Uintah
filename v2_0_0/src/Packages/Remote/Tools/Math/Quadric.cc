// Quadric.cpp - Classes for dealing with 3D Quadrics.
// Peter-Pike Sloan
// Hacked by Dave McAllister, August 1999.

#include <Packages/Remote/Tools/Math/Quadric.h>

namespace Remote {
bool Quadric3::FindMin(Vector& p) const
{
  Matrix44 M(matrix44());
  M(3,0) = 0;
  M(3,1) = 0;
  M(3,2) = 0;
  M(3,3) = 1;
  M.InvalidateInverse();

  bool invertible = M.Invert();

  if(!invertible)
    return false;

  p.x = M(0,3);
  p.y = M(1,3);
  p.z = M(2,3);

  return (finite(p.x) && finite(p.y) && finite(p.z));
}

// Use quadric approximation - infinite lines
/*
   The distance between a Vector and a line is:
   
   Line = o + t*v (o is a Vector, v is a unit vector)

      p  
     /|
   A/ |D - distance - what we are looking for
   o--+---->				    
     B   v (unit length)                    
      projection

  B = Dot((p-o),v)

  Vector for B is o + Dot(p-o,v)*v

  Distance^2 is Dot(p - B, p - B)

  Dot(p-o,v) = px*vx + py*vy + pz*dz - Dot(o,v)

  Dx = (pI - ((px*vx + py*vy + pz*vz - Dov)*vI + oI))^2
  = (pI - px*vI*vx - py*vI*vy - pz*vI*vz + (Dov*vI - oI))^2
                                               Dx
  // now do all of them at once...

  X^2*((1 - vx^2) + vx*vy + vx*vz) +
  Y^2((vy^2 - 1) + vx*vy + vy*vz) +
  Z^2*((1 - vz^2) + vx*vz + vz*vy) +

  // mixed terms...

  XY*(-2*(1 - vx^2)*vx*vy - 2*(1 - vy^2)*vx*vy + 2*vz^2*vx*vy)

  XY*2*vx*vy*(vz^2 - (1 -vx^2) - (1 -vy^2 - 1)) +
  XZ*2*vx*vz*(vy^2 - (1 -vx^2) - (1 -vz^2 - 1)) +
  YZ*2*vy*vz*(vx^2 - (1 -vy^2) - (1 -vx^2 - 1)) +

  // linear terms...

  X*2*((1 -vx^2)*Dx - (vx*vy)*Dy - (vx*vz)*Dz) + 
  Y*2*((1 -vy^2)*Dy - (vx*vy)*Dx - (vy*vz)*Dz) +
  Z*2*((1 -vz^2)*Dz - (vz*vx)*Dx - (vz*vy)*Dy) +
  Dx^2 + Dy^2 + Dz^2
*/

void Quadric3::CreateLine(const Vector &o, const Vector &p1)
{
  Vector v = p1-o;
  v.normalize();

  // Compute the quadric for this guy.

  double Dov = Dot(o,v);
  
  double A,B,C,D,E,F,G,H,I,J;
  
  double vx2m1 = 1 - v.x*v.x;
  double vy2m1 = 1 - v.y*v.y;
  double vz2m1 = 1 - v.z*v.z;
  
  double vxvy = v.x*v.y;
  double vxvz = v.x*v.z;
  double vyvz = v.y*v.z;
  
  double Dx = Dov*v.x - o.x;
  double Dy = Dov*v.y - o.y;
  double Dz = Dov*v.z - o.z;

  double mixedMid = v.x*v.x + v.y*v.y + v.z*v.z - 2;

  A = vx2m1*vx2m1 + vxvy*vxvy + vxvz*vxvz; // X^2
  B = vy2m1*vy2m1 + vxvy*vxvy + vyvz*vyvz; // Y^2
  C = vz2m1*vz2m1 + vxvz*vxvz + vyvz*vyvz; // Z^2
  
#if 0
  D = 2*vxvy*(v.z*v.z - vx2m1 - vy2m1); // XY
  E = 2*vxvz*(v.y*v.y - vx2m1 - vz2m1); // XZ
  F = 2*vyvz*(v.x*v.x - vy2m1 - vz2m1); // YZ
#else
  D = 2*vxvy*mixedMid; // XY
  E = 2*vxvz*mixedMid; // XZ
  F = 2*vyvz*mixedMid; // YZ
#endif  
  G = 2*(vx2m1*Dx - vxvy*Dy - vxvz*Dz); // X
  H = 2*(vy2m1*Dy - vxvy*Dx - vyvz*Dz); // Y
  I = 2*(vz2m1*Dz - vxvz*Dx - vyvz*Dy); // Z
  
  J = Dx*Dx + Dy*Dy + Dz*Dz; // C
  
  // Shove these into the matrix.
  // ^2 coefs are normal, rest are divided by 2.

  vals[0] = A;
  vals[1] = B;
  vals[2] = C;
  vals[3] = J; // constant coef...
  
  vals[4] = D*0.5;
  vals[5] = E*0.5;
  vals[6] = G*0.5;

  vals[7] = F*0.5;
  vals[8] = H*0.5;
  vals[9] = I*0.5;
}
} // End namespace Remote


