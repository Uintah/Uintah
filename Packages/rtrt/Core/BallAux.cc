/***** BallAux.c *****/
#include <math.h>
#include <Packages/rtrt/Core/BallAux.h>

using namespace rtrt;

namespace rtrt {
  Quat qOne(0, 0, 0, 1);
} // end namespace rtrt

/* Construct rotation matrix from (possibly non-unit) quaternion.
 * Assumes matrix is used to multiply column vector on the left:
 * vnew = mat vold.  Works correctly for right-handed coordinate system
 * and right-handed rotations. */

void Quat::ToMatrix(HMatrix& out)
{
    double Nq = x*x + y*y + z*z + w*w;
    double s = (Nq > 0.0) ? (2.0 / Nq) : 0.0;
    double xs = x*s,	      ys = y*s,	    	  zs = z*s;
    double wx = w*xs,	      wy = w*ys,	  wz = w*zs;
    double xx = x*xs,	      xy = x*ys,	  xz = x*zs;
    double yy = y*ys,	      yz = y*zs,	  zz = z*zs;
    out[X][X] = 1.0 - (yy + zz); out[Y][X] = xy + wz; out[Z][X] = xz - wy;
    out[X][Y] = xy - wz; out[Y][Y] = 1.0 - (xx + zz); out[Z][Y] = yz + wx;
    out[X][Z] = xz + wy; out[Y][Z] = yz - wx; out[Z][Z] = 1.0 - (xx + yy);
    out[X][W] = out[Y][W] = out[Z][W] = out[W][X] = out[W][Y] = out[W][Z] = 0.0;
    out[W][W] = 1.0;

}


/* Construct a unit quaternion from rotation matrix.  Assumes matrix is
 * used to multiply column vector on the left: vnew = mat vold.	 Works
 * correctly for right-handed coordinate system and right-handed rotations.
 * Translation and perspective components ignored. */
void Quat::FromMatrix(HMatrix& mat)
{
    /* This algorithm avoids near-zero divides by looking for a large component
     * - first w, then x, y, or z.  When the trace is greater than zero,
     * |w| is greater than 1/2, which is as small as a largest component can be.
     * Otherwise, the largest diagonal entry corresponds to the largest of |x|,
     * |y|, or |z|, one of which must be larger than |w|, and at least 1/2. */
    //Quat qu;
    register double tr, s;

    tr = mat[X][X] + mat[Y][Y]+ mat[Z][Z];
    if (tr >= 0.0) {
	    s = sqrt(tr + mat[W][W]);
	    w = s*0.5;
	    s = 0.5 / s;
	    x = (mat[Z][Y] - mat[Y][Z]) * s;
	    y = (mat[X][Z] - mat[Z][X]) * s;
	    z = (mat[Y][X] - mat[X][Y]) * s;
	} else {
	    int h = X;
	    if (mat[Y][Y] > mat[X][X]) h = Y;
	    if (mat[Z][Z] > mat[h][h]) h = Z;
	    switch (h) {
#define caseMacro(i,j,k,I,J,K) \
	    case I:\
		s = sqrt( (mat[I][I] - (mat[J][J]+mat[K][K])) + mat[W][W] );\
		i = s*0.5;\
		s = 0.5 / s;\
		j = (mat[I][J] + mat[J][I]) * s;\
		k = (mat[K][I] + mat[I][K]) * s;\
		w = (mat[K][J] - mat[J][K]) * s;\
		break
	    caseMacro(x,y,z,X,Y,Z);
	    caseMacro(y,z,x,Y,Z,X);
	    caseMacro(z,x,y,Z,X,Y);
	    }
	}
    if (mat[W][W] != 1.0) *this = V3_Scale(*this, 1/sqrt(mat[W][W]));
//    return (qu);
}

/* Return vector formed from components */
HVect V3_(double x, double y, double z)
{
    HVect v;
    v.x = x; v.y = y; v.z = z; v.w = 0;
    return (v);
}

/* Return norm of v, defined as sum of squares of components */
double V3_Norm(HVect v)
{
    return ( v.x*v.x + v.y*v.y + v.z*v.z );
}

/* Return unit magnitude vector in direction of v */
HVect V3_Unit(HVect v)
{
    static HVect u(0, 0, 0, 0);
    double vlen = sqrt(V3_Norm(v));
    if (vlen != 0.0) {
	u.x = v.x/vlen; u.y = v.y/vlen; u.z = v.z/vlen;
    }
    return (u);
}

/* Return version of v scaled by s */
HVect V3_Scale(HVect v, double s)
{
    HVect u;
    u.x = s*v.x; u.y = s*v.y; u.z = s*v.z; u.w = v.w;
    return (u);
}

/* Return negative of v */
HVect V3_Negate(HVect v)
{
    static HVect u(0, 0, 0, 0);
    u.x = -v.x; u.y = -v.y; u.z = -v.z;
    return (u);
}

/* Return sum of v1 and v2 */
HVect V3_Add(HVect v1, HVect v2)
{
    static HVect v(0, 0, 0, 0);
    v.x = v1.x+v2.x; v.y = v1.y+v2.y; v.z = v1.z+v2.z;
    return (v);
}

/* Return difference of v1 minus v2 */
HVect V3_Sub(HVect v1, HVect v2)
{
    static HVect v(0, 0, 0, 0);
    v.x = v1.x-v2.x; v.y = v1.y-v2.y; v.z = v1.z-v2.z;
    return (v);
}

/* Halve arc between unit vectors v0 and v1. */
HVect V3_Bisect(HVect v0, HVect v1)
{
    HVect v(0, 0, 0, 0);
    double Nv;
    v = V3_Add(v0, v1);
    Nv = V3_Norm(v);
    if (Nv < 1.0e-5) {
	v = V3_(0, 0, 1);
    } else {
	v = V3_Scale(v, 1/sqrt(Nv));
    }
    return (v);
}

/* Return dot product of v1 and v2 */
double V3_Dot(HVect v1, HVect v2)
{
    return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}


/* Return cross product, v1 x v2 */
HVect V3_Cross(HVect v1, HVect v2)
{
    static HVect v(0, 0, 0, 0);
    v.x = v1.y*v2.z-v1.z*v2.y;
    v.y = v1.z*v2.x-v1.x*v2.z;
    v.z = v1.x*v2.y-v1.y*v2.x;
    return (v);
}
