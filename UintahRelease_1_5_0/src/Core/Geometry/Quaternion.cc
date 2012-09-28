/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


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

#include <Core/Geometry/Quaternion.h>

#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Util/Assert.h>

#include <iostream>
using std::cout;
using std::endl;

namespace SCIRun {

#define Abs(x) ((x>0)?x:-x)
#define MSG(m) cout << m << endl;

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
    int mi=0, ia = -1, ix = -1, iy = -1, iz = -1; // -1 quites compiler warning.
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

#define VIEW_VERSION 1

  /* To be implemented
void Pio(Piostream& stream, Quaternion& q){
  
  stream.begin_class("Quaternion", VIEW_VERSION);
  Pio(stream, q.a);
  Pio(stream, q.v);
  stream.end_class();
}
 */

Quaternion Pow(const Quaternion& q, double p){  
  if (q.v.length()> NUM_ZERO){
    double theta=p*acos(q.a);
    return Quaternion(cos(theta), (q.v).normal()*sin(theta));
  }
  else                        // we have reference frame 
    return Quaternion();
}

Quaternion Slerp(const Quaternion& lq, const Quaternion& rq, double h){
  return lq*Pow((lq.get_inv())*rq, h);
}

std::ostream& operator<<(std::ostream& out , const Quaternion& q){
  out << "a=" << q.a << std::endl;
  out << "v=" << q.v << std::endl;
  return out;
}


void test_quat(){

  MSG("Quaternion tests:-----------------------");
  // ****************** TEST 1 ***************************
  MSG("Test#1: creating quaternions");
  
  Quaternion q1;
  MSG("Default Construction:");
  MSG(q1);

  Quaternion q2(Vector(1, 0, 0), Vector(0, 1, 0));
  Quaternion q3(Vector(1, 1, 1), Vector(0, 1, 0));
  MSG("Quaternion q2(Vector(1, 0, 0), Vector(0, 1, 0));");
  MSG(q2);
  MSG("Quaternion q3(Vector(1, 1, 1), Vector(0, 1, 0));");
  MSG(q3);

  Quaternion q4(Vector(2, 2, 2));
  MSG("Quaternion q4(Vector(2, 2, 2))");
  MSG(q4);

  MSG("Norms of the created quaternions:");
  MSG("q1:");
  MSG(q1.norm());
  MSG("q2:");
  MSG(q2.norm());
  MSG("q3:");
  MSG(q3.norm());
  MSG("q4:");
  MSG(q4.norm());
  MSG(" ");

  // ****************** TEST 2 ***************************
  MSG("Test#2: transforms from and to rotational matrix");
  
  MSG(" From and back to q2:");
  MSG(q2);
  Transform rot1;
  q2.to_matrix(rot1);

  MSG("  Created rotational matrix:");
  rot1.print();
  
   Quaternion q2b;
  q2b.from_matrix(rot1);
  MSG("  q2b restored from the matrix:");
  MSG(q2b);
  
  MSG(" Testing construction from prebuilt rotational matrix:");
  MSG("  Corresponding to rotation around OX on -90");
  Transform rot2(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 0, -1), Vector(0, 1, 0));
  rot2.print();
  Quaternion q5(rot2);
  MSG(" Quaternion q5 from the matrix is: " );
  MSG(q5);
  MSG(" Back to the matrix:");
  q5.to_matrix(rot1);
  rot1.print();
  MSG(" ");

   // ****************** TEST 3 ***************************
  MSG("Test#3: operations on quaternions");
  MSG(" Rotating by q5:");
  MSG("  of Vector(1, 1, 1):");
  MSG(q5.rotate(Vector(1, 1, 1)));
  MSG("  of Vector(1, 0, 0):");
  MSG(q5.rotate(Vector(1, 0, 0)));
  MSG("  of Vector(0, 0, 1):");
  MSG(q5.rotate(Vector(0, 0, 1)));
  
  q5=Quaternion();
  MSG(" Rotating of Vector(0, 0, 1) by q equal:");
  MSG(q5);
  MSG(q5.rotate(Vector(0, 0, 1)));

  MSG(" ");
  MSG(" Operator ==:");
  MSG(" q3:"); MSG(q3);
  MSG(" q4:"); MSG(q4);
  MSG(" q3==q4"); MSG((q3==q4));
  MSG(" q3==q3"); MSG((q3==q3));
  MSG(" ");
  MSG(" Quaternion conjugate to q3:");
  MSG(q3.get_inv());
  MSG(" Inversion of q3 itself:");
  q3.inverse();
  MSG(q3);
  MSG(" ");
  MSG(" Sum of quaternions q1 and q5:");
  MSG(" q1:"); MSG(q1);
  MSG(" q5:"); MSG(q5);
  MSG(" Sum Q1+Q5:"); MSG(q1+q5);
  MSG(" ");
  MSG(" Testing get_frame():");
  MSG(" Rot matrix:");
  rot2.print();
  Quaternion q6(rot2);
  MSG( " q6 from the matrix:");
  MSG(q6);
  Vector i, j, k;
  q6.get_frame(i, j, k);
  MSG(" Vectors got from q6:");
  MSG(" i="); MSG(i);
  MSG(" j="); MSG(j);
  MSG(" k="); MSG(k);

  MSG(" ");
  MSG(" Quaternion multiplication:");
  MSG(" Rot matrices of two consequentive rotaitions:");
  rot1.print();
  rot2.print();
  q1.from_matrix(rot1);
  q2.from_matrix(rot2);

  MSG(" Product of the q1 and q2:");
  Quaternion q12=q1*q2;
  MSG(q12);
  
  MSG(" Matrix from the product:");
  Transform rot12;
  q12.to_matrix(rot12);
  rot12.print();
 
 
  
  // ****************** TEST 4 ***************************
  MSG("Test#4: testing Slerp");
  q1=Quaternion(Vector(1, 0, 0), Vector(0, -1, 0));
  MSG(" q1=");
  MSG(q1);
  q2=Quaternion(Vector(0, 0, 1), Vector(0, -1, 0));
  MSG(" q=");
  MSG(q2);
  q3=Slerp(q1, q2, 0);
  MSG(" Slerp(q1, q2, 0):");
  MSG(q3);
  q3=Slerp(q1, q2, 1);
  MSG(" Slerp(q1, q2, 1):");
  MSG(q3);
  q3=Slerp(q1, q2, 0.5);
  MSG(" Slerp(q1, q2, 0.5):");
  MSG(q3);
  
  double ang=Abs(Dot(q2, q1)), w;
  MSG("Angle between q2 and q1 is:");
  MSG(ang);
  for (int i=0; i<=10; i++){ 
    w=(double)i/(double)10;
    MSG("w=:");
    MSG(w);
    q3=Slerp(q1, q2, w);
    MSG(" q3=");
    MSG(q3);  
  }
}

} // End namespace SCIRun



