/*
 * Classes for dealing with different types
 * of quadrics
 * Peter-Pike Sloan
 */

#ifndef _QUADRIC_H_
#define _QUADRIC_H_ 1

#include <Geometry/Point.h>

// 3dimensional quadrics (x,y,z) generaly

struct Quadric3d {
  double vals[10];

  inline Quadric3d operator + (Quadric3d& q);

  inline Quadric3d operator += (Quadric3d& q);

  void zero() { for(int i=0;i<10;i++) vals[i] = 0.0; };

  inline void DoSym(double A, double B, double C, double D);

  // below creates a infinite line...
  void CreateLine(Point &p0, Point &p1);

  inline double MulPt(Point& p); // PQP

  inline void MulVec(Point& p, double res[4]); // QP

  inline void Scale(double v); // now quadric is v PQP instead...

  int FindMin(Point& p); // returns true if found a min
};

struct Quadric4d {
  double vals[15]; // 15 doubles for the full quadric...

  inline Quadric4d operator + (Quadric4d& q);

  inline Quadric4d operator += (Quadric4d& q);

  void zero() { for(int i=0;i<15;i++) vals[i] = 0.0; };

  inline void DoSym(double A, double B, double C, double D, double E);

  inline double MulPt(Point& p, double &v);

  inline void MulVec(Point& p, double &v, double res[5]);

  int FindMin(Point& p, double& v); // returns true if found a min
};

inline Quadric4d  Quadric4d::operator + (Quadric4d& q) 
{
  Quadric4d rval;
  for(int i=0;i<15;i++) rval.vals[i] = vals[i] + q.vals[i];
  return rval; // add 2 Quadric4ds...
}

inline Quadric4d Quadric4d::operator += (Quadric4d& q) 
{
  for(int i=0;i<15;i++) vals[i] += q.vals[i];
  return *this;
}

void Quadric4d::DoSym(double A, double B, double C, double D, double E) 
{
  vals[0] = A*A;
  vals[1] = B*B;
  vals[2] = C*C; 
  vals[3] = D*D;
  vals[4] = E*E;
  
  vals[5] = A*B;
  vals[6] = A*C;
  vals[7] = A*D;
  vals[8] = A*E;
  
  vals[9] = B*C;
  vals[10] = B*D;
  vals[11] = B*E;
  
  vals[12] = C*D;
  vals[13] = C*E;
  
  vals[14] = D*E;
}

  // matrix looks like this (w/respect to indeces)

  /*
     	+--+--+--+--+--+
	+ 0+ 5+ 6+ 7+ 8+
     	+--+--+--+--+--+
	+ 5+ 1+ 9+10+11+
     	+--+--+--+--+--+
	+ 6+ 9+ 2+12+13+
     	+--+--+--+--+--+
	+ 7+10+12+ 3+14+
     	+--+--+--+--+--+
	+ 8+11+13+14+ 4+
     	+--+--+--+--+--+
   */


inline double Quadric4d::MulPt(Point& p, double &v) { // compute PQP
  
  return ( (p.x()*(p.x()*vals[0] + p.y()*vals[5] + p.z()*vals[6] + v*vals[7] + vals[8]))
	  + (p.y()*(p.x()*vals[5] + p.y()*vals[1] + p.z()*vals[9] + v*vals[10] + vals[11]))
	  +(p.z()*(p.x()*vals[6] + p.y()*vals[9] + p.z()*vals[2] + v*vals[12] + vals[13]))
	  +v*(p.x()*vals[7] + p.y()*vals[10] + p.z()*vals[12] + v*vals[3] + vals[14])
	  +(p.x()*vals[8] + p.y()*vals[11] + p.z()*vals[13] + v*vals[14] + vals[4]) );
}

inline void Quadric4d::MulVec(Point &p, double &v, double res[5]) { // QP
  res[0] = (p.x()*vals[0] + p.y()*vals[5] + p.z()*vals[6] + v*vals[7] + vals[8]);
  res[1] = (p.x()*vals[5] + p.y()*vals[1] + p.z()*vals[9] + v*vals[10] + vals[11]);
  res[2] = (p.x()*vals[6] + p.y()*vals[9] + p.z()*vals[2] + v*vals[12] + vals[13]);
  res[3] = (p.x()*vals[7] + p.y()*vals[10] + p.z()*vals[12] + v*vals[3] + vals[14]);
  res[4] = (p.x()*vals[8] + p.y()*vals[11] + p.z()*vals[13] + v*vals[14] + vals[4]);
}


inline Quadric3d  Quadric3d::operator + (Quadric3d& q) {
  Quadric3d rval;
  for(int i=0;i<10;i++) rval.vals[i] = vals[i] + q.vals[i];
  return rval; // add 2 Quadric3ds...
}

inline Quadric3d Quadric3d::operator += (Quadric3d& q) {
  for(int i=0;i<10;i++) vals[i] += q.vals[i];
  return *this;
}

  // matrix looks like this (w/respect to indeces)

  /*
        +--+--+--+--+   
        + 0+ 4+ 5+ 6+   
     	+--+--+--+--+   
        + 4+ 1+ 7+ 8+   
     	+--+--+--+--+   
	+ 5+ 7+ 2+ 9+   
     	+--+--+--+--+   
	+ 6+ 8+ 9+ 3+   
     	+--+--+--+--+   

	X^2 -> 0
	Y^2 -> 1
	Z^2 -> 2
	C   -> 3

	XY  -> 4
	XZ  -> 5
	X   -> 6
	YZ  -> 7
	Y   -> 8
	Z   -> 9
		 
   */		 
		 
void Quadric3d::DoSym(double A, double B, double C, double D) {
  vals[0] = A*A; 
  vals[1] = B*B; 
  vals[2] = C*C; 
  vals[3] = D*D; 
  		 
  vals[4] = A*B; 
  vals[5] = A*C;
  vals[6] = A*D;
  
  vals[7] = B*C;
  vals[8] = B*D;
  
  vals[9] = C*D;
}



inline double Quadric3d::MulPt(Point& p) { // compute PQP
#if 0  
  return ( (p.x()*(p.x()*vals[0] + p.y()*vals[4] + p.z()*vals[5] + vals[6]))
	  + (p.y()*(p.x()*vals[4] + p.y()*vals[1] + p.z()*vals[7] + vals[8]))
	  +(p.z()*(p.x()*vals[5] + p.y()*vals[7] + p.z()*vals[2] + vals[9]))
	  +(p.x()*vals[6] + p.y()*vals[8] + p.z()*vals[9] + vals[3]) );
#else
  return ( (p.x()*p.x()*vals[0] + p.y()*p.y()*vals[1] + p.z()*p.z()*vals[2] + vals[3] 
	    + 2*p.x()*p.y()*vals[4] + 2*p.x()*p.z()*vals[5] + 2*p.y()*p.z()*vals[7] +
	    2*p.x()*vals[6] + 2*p.y()*vals[8] + 2*p.z()*vals[9]) );
#endif
}

// using above equation, squared teams multiplied by v, rest by v/2

inline void Quadric3d::Scale(double v) {
  for(int i=0;i<3;i++) {
    vals[i] *= v;
  }
  v = v*0.5; // mixed and single stuff...
  for(;i<10;i++) {
    vals[i] *= v;
  }
}

inline void Quadric3d::MulVec(Point &p, double res[4]) { // QP
  res[0] = (p.x()*vals[0] + p.y()*vals[4] + p.z()*vals[5] + vals[6]);
  res[1] = (p.x()*vals[4] + p.y()*vals[1] + p.z()*vals[7] + vals[8]);
  res[2] = (p.x()*vals[5] + p.y()*vals[7] + p.z()*vals[2] + vals[9]);
  res[3] = (p.x()*vals[6] + p.y()*vals[8] + p.z()*vals[9] + vals[3]);
}

#endif
