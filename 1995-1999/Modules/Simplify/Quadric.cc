/*
 * Code for dealing with quadrics
 * Peter-Pike Sloan
 */

#include "Quadric.h"
#include <iostream.h>

// functions for 3D quadrics...

int InvertMat(double mat[4][4], double rhs[4], Point& pt)
{
#if 0
  double s[4];   // row weights
  int    p[4];   // permutation index

  
  for(int i=0;i<4;i++) { // initialize stuff
    p[i] = i;
    s[i] = fabs(mat[i][0]); // just set it to the first coord...
    for(int j=1;j<4;j++) {
      if (fabs(mat[i][j]) > s[i])
	s[i] = fabs(mat[i][j]);
    }
  }

  // now start the proccess...

  for(int k=0;k<3;k++) {
    int cur_p = k; 
    double cur_w = fabs(mat[p[cur_p]][k])/s[p[cur_p]];
    for(int j=k+1;j<4;j++) { // find which row to use...
      double tw = fabs(mat[p[j]][k])/s[p[j]];
      if (tw > cur_w) {
	cur_p = j;
	cur_w = tw;
      }
    }

    // now swap in the pivot row...

    int tmp=p[k];
    p[k] = p[cur_p];
    p[cur_p] = tmp;

    if (fabs(mat[p[k]][k]) < 0.0000001) {
      // any zeroes as a pivot means it isn't invertable...
      cerr << k << " - Can't invert!\n";
      return 0;
    }

    double one_over = 1.0/mat[p[k]][k];
    
    for(i=k+1;i<4;i++) {
      double z = mat[p[i]][k]*one_over;
      
      for(j=k+1;j<4;j++) {
	mat[p[i]][j] = mat[p[i]][j] - z*mat[p[k]][j];
      }
      rhs[p[i]] = rhs[p[i]] - z*rhs[p[k]]; // update rhs as well...
    }
  }

  // now do the back substitution...

  rhs[p[3]] = rhs[p[3]]/mat[p[3]][3];

  for(k=2;k>=0;k--) {
    double val=0.0;
    for(int i=k+1;i<4;i++) {
      if (fabs(mat[p[k]][i]) > 0.000001)
	val += -(rhs[p[i]]/mat[p[k]][i]);
    }
    rhs[p[k]] = (rhs[p[k]] + val)/mat[p[k]][k];
  }

  pt.x(rhs[p[0]]);
  pt.y(rhs[p[1]]);
  pt.z(rhs[p[2]]);

  return 1;
#else

  // stole code from garland/heckbert for this...
#define SWAP(a, b, t)   {t = a; a = b; b = t;}

  int i, j, k;
  double max, t, det, pivot;

  /*---------- forward elimination ----------*/

  double B[4][4];
  double A[4][4];

  for (i=0; i<4; i++)                 /* put identity matrix in B */
    for (j=0; j<4; j++) {
      B[i][j] = (double)(i==j);
      A[i][j] = mat[i][j];
    }

  det = 1.0;
  for (i=0; i<4; i++) {               /* eliminate in column i, below diag */
    max = -1.;
    for (k=i; k<4; k++)             /* find pivot for column i */
      if (fabs(A[k][i]) > max) {
	max = fabs(A[k][i]);
	j = k;
      }
    if (max<=0.) return 0.;         /* if no nonzero pivot, PUNT */
    if (j!=i) {                     /* swap rows i and j */
      for (k=i; k<4; k++)
	SWAP(A[i][k], A[j][k], t);
      for (k=0; k<4; k++)
	SWAP(B[i][k], B[j][k], t);
      det = -det;
    }
    pivot = A[i][i];
    det *= pivot;
    for (k=i+1; k<4; k++)           /* only do elems to right of pivot */
      A[i][k] /= pivot;
    for (k=0; k<4; k++)
      B[i][k] /= pivot;
    /* we know that A(i, i) will be set to 1, so don't bother to do it */
    
    for (j=i+1; j<4; j++) {         /* eliminate in rows below i */
      t = A[j][i];                /* we're gonna zero this guy */
      for (k=i+1; k<4; k++)       /* subtract scaled row i from row j */
	A[j][k] -= A[i][k]*t;   /* (ignore k<=i, we know they're 0) */
      for (k=0; k<4; k++)
	B[j][k] -= B[i][k]*t;
    }
  }

  /*---------- backward elimination ----------*/

  for (i=4-1; i>0; i--) {             /* eliminate in column i, above diag */
    for (j=0; j<i; j++) {           /* eliminate in rows above i */
      t = A[j][i];                /* we're gonna zero this guy */
      for (k=0; k<4; k++)         /* subtract scaled row i from row j */
	B[j][k] -= B[i][k]*t;
    }
  }

  pt.x(B[0][3]);
  pt.y(B[1][3]);
  pt.z(B[2][3]);
  
#endif
}


static int Q3dremap[4][4] = { {0,4,5,6},
			      {4,1,7,8},
			      {5,7,2,9},
			      {6,8,9,3} };

int Quadric3d::FindMin(Point& p)
{
  // first build the matrix we need to hand off...

  double mat[4][4];
  for(int j=0;j<3;j++)
    for(int i=0;i<4;i++) {
      mat[j][i] = vals[Q3dremap[j][i]];
    }
  for(int i=0;i<3;i++)
    mat[j][i] = 0.0;
  mat[j][i] = 1.0;

  double rhs[4] = {0,0,0,1.0};  // try and invert this point...

  // now we just need to hand this off to a solver that can invert
  // a matrix.  If the matrix is not invertable - what would the
  // pseudo inverse give us???

  return InvertMat(mat,rhs,p);
}


// below are functions for Quadric4d...

void MatPrint(double m[4][4])
{
  cerr << "Matrix: \n";
  for(int j=0;j<4;j++) {
    for(int i=0;i<4;i++) {
      cerr << m[j][i] << " ";
    }
    cerr << "\n";
  }
}

// solves a small 4x4 matrix, 

void SolveMat(double mat[4][4], int &x, int &y, int &z, int &w,
	      double vec[4]) // this vector is the "normal" in that space
{
  double s[4]; // scales - for pivoting
  int    p[4]; // permutation, for pivoting...

  for(int i=0;i<4;i++) { // initialize stuff
    p[i] = i;
    s[i] = fabs(mat[i][0]); // just set it to the first coord...
    for(int j=1;j<4;j++) {
      if (fabs(mat[i][j]) > s[i])
	s[i] = fabs(mat[i][j]);
    }
  }

  // now start the proccess...

  for(int k=0;k<3;k++) {
    int cur_p = k; 
    double cur_w = fabs(mat[p[cur_p]][k])/s[p[cur_p]];
    for(int j=k+1;j<4;j++) { // find which row to use...
      double tw = fabs(mat[p[j]][k])/s[p[j]];
      if (tw > cur_w) {
	cur_p = j;
	cur_w = tw;
      }
    }

    // now swap in the pivot row...

    int tmp=p[k];
    p[k] = p[cur_p];
    p[cur_p] = tmp;

    if (fabs(mat[p[k]][k]) < 0.0000001) {
      while (k<4) {
	int j=k;
	while(j<4)
	  mat[p[k]][j++] = 0.0;
	k++;
      }
      z=1; // set the extra flag...
      break;
    }

    double one_over = 1.0/mat[p[k]][k];
    
    for(i=k+1;i<4;i++) {
      double z = mat[p[i]][k]*one_over;
      
      for(j=k+1;j<4;j++) {
	mat[p[i]][j] = mat[p[i]][j] - z*mat[p[k]][j];
      }
    }
  }

  // double check the last guy - see if it is a zero...

  if (fabs(mat[p[3]][3]) > 0.000001) {
    cerr << mat[p[3]][3] << " Not Degenerate?\n";
  }

  // ok, now just do back substitution to get the vector...

  vec[3] = 1.0; // just assign the last guy for now...

  vec[2] = -mat[p[2]][3]/mat[p[2]][2]; // should have been last pivot...

  vec[1] = (-vec[3]*mat[p[1]][3] - vec[2]*mat[p[1]][2])/mat[p[1]][1];

  vec[0] = (-vec[3]*mat[p[0]][3] - vec[2]*mat[p[0]][2] -
	    vec[1]*mat[p[0]][1])/mat[p[0]][0];

  // back substitution is finished, normalize this vector...

  double mag = sqrt(vec[0]*vec[0] + vec[1]*vec[1] +
		    vec[2]*vec[2] + vec[3]*vec[3]);

  mag = 10.0/mag; // push this through the normal... -> scale up...

  for(k=0;k<4;k++) {
    vec[k] *= mag;
    if (fabs(vec[k]) < 0.000001) {
//      vec[k] = 0.0; // just zero it out...
    }
  }

  w=1; // set it up...
}

int InvertMat(double mat[5][5], double rhs[5], Point& pt, double& v)
{
  double s[5];   // row weights
  int    p[5];   // permutation index

  
  for(int i=0;i<5;i++) { // initialize stuff
    p[i] = i;
    s[i] = fabs(mat[i][0]); // just set it to the first coord...
    for(int j=1;j<5;j++) {
      if (fabs(mat[i][j]) > s[i])
	s[i] = fabs(mat[i][j]);
    }
  }

  // now start the proccess...

  for(int k=0;k<4;k++) {
    int cur_p = k; 
    double cur_w = fabs(mat[p[cur_p]][k])/s[p[cur_p]];
    for(int j=k+1;j<5;j++) { // find which row to use...
      double tw = fabs(mat[p[j]][k])/s[p[j]];
      if (tw > cur_w) {
	cur_p = j;
	cur_w = tw;
      }
    }

    // now swap in the pivot row...

    int tmp=p[k];
    p[k] = p[cur_p];
    p[cur_p] = tmp;

    if (fabs(mat[p[k]][k]) < 0.0000001) {
      // any zeroes as a pivot means it isn't invertable...
      cerr << k << " - Can't invert!\n";
      return 0;
    }

    double one_over = 1.0/mat[p[k]][k];
    
    for(i=k+1;i<5;i++) {
      double z = mat[p[i]][k]*one_over;
      
      for(j=k+1;j<5;j++) {
	mat[p[i]][j] = mat[p[i]][j] - z*mat[p[k]][j];
      }
      rhs[p[i]] = rhs[p[i]] - z*rhs[p[k]]; // update rhs as well...
    }
  }

  // now do the back substitution...

  rhs[p[4]] = rhs[p[4]]/mat[p[4]][4];

  for(k=3;k>=0;k--) {
    double val=0.0;
    for(int i=k+1;i<5;i++) {
      if (fabs(mat[p[k]][i]) > 0.000001)
	val += -(rhs[p[i]]/mat[p[k]][i]);
    }
    rhs[p[k]] = (rhs[p[k]] + val)/mat[p[k]][k];
  }

  pt.x(rhs[p[0]]);
  pt.y(rhs[p[1]]);
  pt.z(rhs[p[2]]);
  v = rhs[p[3]]; // fill in this minimum guy...

  return 1;
}

static int Q4dremap[5][5] = { {0,5,6,7,8},
			      {5,1,9,10,11},
			      {6,9,2,12,13},
			      {7,10,12,3,14},
			      {8,11,13,14,4} };

int Quadric4d::FindMin(Point& p, double& v)
{
  // first build the matrix we need to hand off...

  double mat[5][5];
  for(int j=0;j<4;j++)
    for(int i=0;i<5;i++) {
      mat[j][i] = vals[Q4dremap[j][i]];
    }
  for(int i=0;i<4;i++)
    mat[j][i] = 0.0;
  mat[j][i] = 1.0;

  double rhs[5] = {0,0,0,0,1.0};  // try and invert this point...

  // now we just need to hand this off to a solver that can invert
  // a matrix.  If the matrix is not invertable - what would the
  // pseudo inverse give us???

  return InvertMat(mat,rhs,p,v);
}

// this function computes the actual quadric associated
// with the given points and scalar values...

void ComputeQuadric(Quadric4d& res, 
		    Point& p0,Point& p1,Point& p2,Point& p3,
		    double v0, double v1, double v2, double v3)
{
  double mat[4][4]; // matrix that will be reduced...

  Point rp0 = (p0.vector() + p1.vector() - p2.vector() - p3.vector()).point();
  Point rp1 = (p0.vector() - p1.vector() + p2.vector() - p3.vector()).point();
  Point rp2 = (p0.vector() -p1.vector() - p2.vector() + p3.vector()).point();
  Point rp3 = (p0.vector()*3 -p1.vector()*2 - p2.vector()*2 + p3.vector()).point();

//  Point rp3 = (- p0.vector() +p1.vector() - p2.vector() + p3.vector()).point();
  double rv0 = v0 + v1 - v2 - v3;
  double rv1 = v0 - v1 + v2 - v3;
  double rv2 = v0 - v1 - v2 + v3;
  double rv3 = 3*v0 - v1*2 - v2*2 + v3;

  Point rp[4] = {rp0,rp1,rp2,rp3};
  double rv[4] = {rv0,rv1,rv2,rv3};


  for(int j=0;j<4;j++) {
    mat[j][0] = rp[j].x();
    mat[j][1] = rp[j].y();
    mat[j][2] = rp[j].z();
    mat[j][3] = rv[j];
  }

  int x=0,y=0,z=0,s=0; // set if they are "free"

  double vec[4]; // normal vector for the plane equation...

  SolveMat(mat,x,y,z,s,vec);

  if (x+y+z+s > 1) {
    res.zero();
    cerr << "Degenerate Quadric!\n";
  }  else if ((x+y+z+s) == 0) {
    cerr << "Should only be 3 DOF!\n";
    res.zero();
  } else { // it was ok...
    double E = -(p0.x()*vec[0] + p0.y()*vec[1] + p0.z()*vec[2] +
      v0*vec[3]);

    double E1 = -(p1.x()*vec[0] + p1.y()*vec[1] + p1.z()*vec[2] +
      v1*vec[3]);
    double E2 = -(p2.x()*vec[0] + p2.y()*vec[1] + p2.z()*vec[2] +
      v2*vec[3]);
    double E3 = -(p3.x()*vec[0] + p3.y()*vec[1] + p3.z()*vec[2] +
      v3*vec[3]);

    double A,B,C,D; // these are the quadric coefs

    if (fabs(E) < 0.000001) {
//      E = 0.0; // just zero it out...
//      cerr << "Small E...";
    }

    A = vec[0];B = vec[1]; C=vec[2]; D = vec[3];
    res.DoSym(A,B,C,D,E); // fills in the rest...

    Point ptest(-0.5,0,0.8);
    double vtest=1.0;
  }
  
}
// use quadric approximation - infinite lines

/*
   The distance between a point and a line is:
   
   Line = o + t*v (o is a point, v is a unit vector)

      p  
     /|
   A/ |D - distance - what we are looing for
   o--+---->				    
     B   v (unit length)                    
      projection

  B = Dot((p-o),v)

  point for B is o + Dot(p-o,v)*v

  Distance^2 is Dot(p - B, p - B)

  Dot(p-o,v) = px*vx + py*vy + pz*dz - Dot(o,v)

  Dx = (pI - ((px*vx + py*vy + pz*vz - Dov)*vI + oI))^2
  = (pI - px*vI*vx - py*vI*vy - pz*vI*vz + (Dov*vI - oI))^2
                                               Dx
  // now do all of them at once...

  X^2*((1 - vx^2) + vx*vy + vx*vz) + Y^2((vy^2 - 1) + vx*vy + vy*vz) +
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

void Quadric3d::CreateLine(Point &o, Point &p1)
{
  Vector v = p1-o;
  v.normalize();

  // now compute the quadric for this guy...

  double Dov = Dot(o,v);
  
  double A,B,C,D,E,F,G,H,I,J;
  
  double vx2m1 = 1 - v.x()*v.x();
  double vy2m1 = 1 - v.y()*v.y();
  double vz2m1 = 1 - v.z()*v.z();
  
  double vxvy = v.x()*v.y();
  double vxvz = v.x()*v.z();
  double vyvz = v.y()*v.z();
  
  double Dx = Dov*v.x() - o.x();
  double Dy = Dov*v.y() - o.y();
  double Dz = Dov*v.z() - o.z();

  double mixedMid = v.x()*v.x() + v.y()*v.y() + v.z()*v.z() - 2;

  A = vx2m1*vx2m1 + vxvy*vxvy + vxvz*vxvz; // X^2
  B = vy2m1*vy2m1 + vxvy*vxvy + vyvz*vyvz; // Y^2
  C = vz2m1*vz2m1 + vxvz*vxvz + vyvz*vyvz; // Z^2
  
#if 0
  D = 2*vxvy*(v.z()*v.z() - vx2m1 - vy2m1); //XY
  E = 2*vxvz*(v.y()*v.y() - vx2m1 - vz2m1); //XZ
  F = 2*vyvz*(v.x()*v.x() - vy2m1 - vz2m1); //YZ
#else
  D = 2*vxvy*mixedMid; //XY
  E = 2*vxvz*mixedMid; //XZ
  F = 2*vyvz*mixedMid; //YZ
#endif  
  G = 2*(vx2m1*Dx - vxvy*Dy - vxvz*Dz); //X
  H = 2*(vy2m1*Dy - vxvy*Dx - vyvz*Dz); //Y
  I = 2*(vz2m1*Dz - vxvz*Dx - vyvz*Dy); //Z
  
  J = Dx*Dx + Dy*Dy + Dz*Dz; //C
  
  // now shove these into the matrix ^2 coefs are
  // normal, rest are divided by 2...

  vals[0] = A;
  vals[1] = B;
  vals[2] = C;
  vals[3] = J; // constant coef...
  
  vals[4] = D/2;
  vals[5] = E/2;
  vals[6] = G/2;

  vals[7] = F/2;
  vals[8] = H/2;
  vals[9] = I/2;

}
