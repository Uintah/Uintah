/*
 *  Quaternion.cc: quaternion representation
 *
 *  Written by:
 *   Author: Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   Date: July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/Geometry/Quaternion.h>
#include <SCICore/share/share.h>

#include <SCICore/Geometry/Transform.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Util/Assert.h>

#include <iostream>
using std::cout;
using std::endl;

namespace SCICore {
namespace Geometry {

#define Abs(x) ((x>0)?x:-x)

//****************************************************************************************
// construction/destruction

// construct quaternion from two vectors, representing i and j axis of some frame
Quaternion::Quaternion(Vector i, Vector u){
  if (i.length()> NUM_ZERO && u.length() > NUM_ZERO){
    i.normalize();
    u.normalize();
    if (Abs(Dot(i, u)) > 0.9999){ // almost paralel
      v=i;
      a=0;
    }
    else {
      Vector k=Cross(i, u);
      Vector j=Cross(k, i);
      k.normalize();
      j.normalize();
      Transform rm;
      rm.load_frame(Point(0,0,0), i, j, k);
      from_matrix(rm);
    }
  }
  else{
    a=1;
    v=Vector(0, 0, 0);
  }
}
// construct quat. from rotational matrix
Quaternion::Quaternion(const Transform& m){
  from_matrix(m);
}

// quaternion represenation of a vector
Quaternion::Quaternion(const Vector& vect){
  v=vect;
  a=0;
  normalize();
}

void Quaternion::to_matrix(Transform& m){
  double (&matr)[4][4]=m.mat;
  double vx=v.x(), vy=v.y(), vz=v.z();
  matr[0][0] = 1.0 - 2.0*(vy*vy + vz*vz);
  matr[1][0] = 2.0*(vx*vy + vz*a);
  matr[2][0] = 2.0*(vz*vx - vy*a);
  matr[3][0] = 0.0;
  
  matr[0][1] = 2.0*(vx*vy - vz*a);
  matr[1][1] = 1.0 - 2.0*(vz*vz + vx*vx);
  matr[2][1] = 2.0*(vy*vz + vx*a);
  matr[3][1] = 0.0;
  
  matr[0][2] = 2.0*(vz*vx + vy*a);
  matr[1][2] = 2.0*(vy*vz - vx*a);
  matr[2][2] = 1.0 - 2.0*(vy*vy + vx*vx);
  matr[3][2] = 0.0;

  matr[0][3] = 0.0;
  matr[1][3] = 0.0;
  matr[2][3] = 0.0;
  matr[3][3] = 1.0;
}

void Quaternion::from_matrix(const double matr[4][4]){

  double trace=matr[0][0]+matr[1][1]+matr[2][2]+1;  
  double s;
  if (trace>10e-7) {
    // trace is positive
    s=0.5/sqrt(trace);
    a=0.25/s;
    v.x((matr[2][1]-matr[1][2])*s);
    v.y((matr[0][2]-matr[2][0])*s);
    v.z((matr[1][0]-matr[0][1])*s);
    // is_unit=false;
  }
  else {
    //finding max diag element
    int mi=0, ia, ix, iy, iz;
    double qtmp[4];
    for (int i=0; i<=2; i++)
      if (matr[mi][mi]<matr[i][i])
	mi=i;
   
    s=2+2*matr[mi][mi]-trace;
    qtmp[0]=0.5/s;
    qtmp[1]=(matr[0][1]+matr[1][0])/s;
    qtmp[2]=(matr[0][2]+matr[2][0])/s;
    qtmp[3]=(matr[1][2]+matr[2][1])/s;
    switch (mi) {
    case 0:
      ia=3;
      ix=0;
      iy=1;
      iz=2;
      break;
    case 1:
      ia=2;
      ix=1;
      iy=3;
      iz=0;
      
      break;
    case 2:
      ia=1;
      ix=2;
      iy=0;
      iz=3;
      break;
    }
    a=qtmp[ia];
    v.x(qtmp[ix]);
    v.y(qtmp[iy]);
    v.z(qtmp[iz]);
  }
  this->normalize();
}

void Quaternion::get_frame(Vector& i, Vector& j, Vector& k){
  Transform matr;
  this->to_matrix(matr);
  i=Vector(matr.mat[0][0], matr.mat[1][0], matr.mat[2][0]); // ?????
  j=Vector(matr.mat[0][1], matr.mat[1][1], matr.mat[2][1]);
  k=Vector(matr.mat[0][2], matr.mat[1][2], matr.mat[2][2]);
}

bool Quaternion::operator==(const Quaternion& q){
  return (Abs(this->a- q.a)<10e-7  && this->v == q.v);
}

  /*
#define VIEW_VERSION 1

void Pio(Piostream& stream, Quaternion& q){
  using SCICore::PersistentSpace::Pio;
  using SCICore::Geometry::Pio;
  
  stream.begin_class("Quaternion", VIEW_VERSION);
  Pio(stream, q.a);
  Pio(stream, q.v);
  stream.end_class();
}
 */
Quaternion pow(const Quaternion& q, double p){  
  if (q.v.length()> NUM_ZERO){
    double theta=p*acos(q.a);
    return Quaternion(cos(theta), (q.v).normal()*sin(theta));
  }
  else                        // we have reference frame 
    return Quaternion();
}

Quaternion Slerp(const Quaternion& lq, const Quaternion& rq, double h){
  return lq*pow((lq.get_inv())*rq, h);
}

std::ostream& operator<<(std::ostream& out , const Quaternion& q){
  out << "a=" << q.a << std::endl;
  out << "v=" << q.v << std::endl;
  return out;
}

} // Geometry
} // SCICore



