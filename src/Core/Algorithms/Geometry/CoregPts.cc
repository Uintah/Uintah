//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : CoregPts.cc
//    Author : David Weinstein
//    Date   : Wed Dec  5 16:05:07 MST 2001

#include <Core/Algorithms/Geometry/CoregPts.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Trig.h>
#include <iostream>

using std::cerr;

namespace SCIRun {

CoregPts::~CoregPts() {
}

void CoregPts::setOrigPtsA(Array1<Point> a) { 
  origPtsA_ = a; 
  invalidate(); 
}

void CoregPts::setOrigPtsP(Array1<Point> p) { 
  origPtsP_ = p; 
  invalidate(); 
}

int CoregPts::getTransPtsA(Array1<Point> &p) { 
  if (computeTransPtsA()) { 
    p = transPtsA_ ; 
    return 1; 
  } else 
    return 0; 
}

int CoregPts::getTrans(Transform &t) {
  if (computeTrans()) { 
    t = transform_; 
    return 1; 
  } else 
    return 0; 
}

void CoregPts::invalidate() { 
  validTransPtsA_ = 0; 
  validTrans_ = 0; 
}

int CoregPts::computeTransPtsA() {
  if (validTransPtsA_) return 1;
  if (!validTrans_)
    if (!computeTrans()) return 0;
  transPtsA_.resize(0);
  for (int i=0; i<origPtsA_.size(); i++)
    transPtsA_.add(transform_.project(origPtsA_[i]));
  validTransPtsA_ = 1;
  return 1;
}

int CoregPts::getMisfit(double &misfit) {
  if (!computeTransPtsA())
    return 0;
  misfit=0;
  int npts = origPtsP_.size();
  for (int i=0; i<npts; i++)
    misfit += (origPtsP_[i]-transPtsA_[i]).length2()/npts;
  misfit = Sqrt(misfit);
  return 1;
}

CoregPtsAnalytic::~CoregPtsAnalytic() {
}

int CoregPtsAnalytic::computeTrans() {
  unsigned int i;
  if (validTrans_) return 1;

  // make sure we have the right number of points
  if (origPtsA_.size() != 3 || origPtsP_.size() != 3) return 0;

  // make sure the three A points aren't colinear
  Vector a01 = origPtsA_[1] - origPtsA_[0];
  a01.normalize();
  Vector a21 = origPtsA_[1] - origPtsA_[2];
  a21.normalize();
  if (Abs(Dot(a01,a21)) > 0.999) return 0;

  // make sure the three P points aren't colinear
  Vector p01 = origPtsP_[1] - origPtsP_[0];
  p01.normalize();
  Vector p21 = origPtsP_[1] - origPtsP_[2];
  p21.normalize();
  if (Abs(Dot(p01,p21)) > 0.999) return 0;

  // input is valid, let's compute the optimal transform according
  // to Weinstein's Univ of Utah Tech Report UUCS-98-005:
  // "The Analytic 3-D Transform for the Least-Squared Fit of Three
  // Pairs of Corresponding Points" (March, 1998).

  // To tranform the points "orig_a" to "trans_a" such that they best align
  //  with their corresponding points "orig_p", we use this equation:

  // trans_a = (TCp * Bpt * S * Theta * Ba * TC_a) * orig_a

  // Here are the definitions of the intermediate matrices:
  //  TC_a  : Translate vertices to be centered about the origin
  //  Ba    : Rotate the vertices into the xy plane
  //  Theta : Rotate the vertices within the xy plane
  //  S     : Scale the distance from the vertices to the origin
  //  Bpt   : Rotate the vertices from the xy plane to the coordinate from of p
  //  TCp   : Translate the vertices to have the same centroid as p

  Transform TCp, Bpt, S, Theta, Ba, TC_a;
  
  // first, compute the mid-points of a and p
  Point Cp(AffineCombination(origPtsP_[0], 1./3,
			     origPtsP_[1], 1./3,
			     origPtsP_[2], 1./3));
  Point Ca(AffineCombination(origPtsA_[0], 1./3,
			     origPtsA_[1], 1./3,
			     origPtsA_[2], 1./3));
  TC_a.pre_translate(-(Ca.asVector()));
  TCp.pre_translate(Cp.asVector());
  
  // find the normal and tangents for triangle a and for triangle p
  Vector a20=Cross(a01, a21);
  a20.normalize();
  a20.find_orthogonal(a01, a21);

  Vector p20=Cross(p01, p21);
  p20.normalize();
  p20.find_orthogonal(p01, p21);
  
  Transform temp;
  double d[16];
  temp.load_frame(Point(0,0,0), a01, a21, a20);
  temp.get_trans(&(d[0]));
  Ba.set(d);

  Bpt.load_frame(Point(0,0,0), p01, p21, p20);

//  Bpt.load_identity();
//  Ba.load_identity();

  // find optimal rotation theta
  // this is easier if we transform the points through the above transform
  // into their "canonical" position -- triangles centered at the origin,
  // and lying in the xy plane.

  double ra[3], rp[3], theta[3];  
  Point a[3], p[3];

  for (i=0; i<3; i++) {
    // build the canonically-posed vertices
    a[i]=Ba.project(TC_a.project(origPtsA_[i]));
    p[i]=Bpt.unproject(TCp.unproject(origPtsP_[i]));

    // compute their distance from the origin
    ra[i] = a[i].asVector().length();
    rp[i] = p[i].asVector().length();

    // find the angular distance (in radians) between corresponding points
    Vector avn(a[i].asVector());
    avn.normalize();
    Vector pvn(p[i].asVector());
    pvn.normalize();
    theta[i] = Acos(Dot(avn,pvn));

    // make sure we have the sign right
    if (Cross(avn,pvn).z() < 0) theta[i]*=-1;    
  }

  double theta_best = -theta[0] + Atan((ra[1]*rp[1]*Sin(theta[0]-theta[1])+
					ra[2]*rp[2]*Sin(theta[0]-theta[2]))/
				       (ra[0]*rp[0]+
					ra[1]*rp[1]*Cos(theta[0]-theta[1])+
					ra[2]*rp[2]*Cos(theta[0]-theta[2])));
  Theta.pre_rotate(-theta_best, Vector(0,0,1));

  // lastly, rotate the a points into position and solve for scale
  
  Vector av[3];
  double scale_num=0, scale_denom=0;
  for (i=0; i<3; i++) {
    av[i] = Theta.project(a[i]).asVector();
    scale_num += Dot(av[i], p[i].asVector());
    scale_denom += av[i].length2();
  }
  
  double scale = scale_num/scale_denom;
  S.pre_scale(Vector(scale, scale, scale));

  transform_.load_identity();
  transform_.pre_trans(TC_a);
  transform_.pre_trans(Ba);
  transform_.pre_trans(Theta);
  transform_.pre_trans(S);
  transform_.pre_trans(Bpt);
  transform_.pre_trans(TCp);

  validTrans_ = 1;

  // DEBUG
#if 0
  computeTransPtsA();
  double misfit;
  getMisfit(misfit);

  cerr << "Original A points:\n";
  for (i=0; i<3; i++)
    cerr << origPtsA_[i] << "\n";

  cerr << "\nTranslated A points:\n";
  for (i=0; i<3; i++)
    cerr << TC_a.project(origPtsA_[i]) << "\n";

  cerr << "\nxy-plane A points:\n";
  for (i=0; i<3; i++)
    cerr << a[i] << "\n";

  cerr << "\nOriginal P points:\n";
  for (i=0; i<3; i++)
    cerr << origPtsP_[i] << "\n";

  cerr << "\nUn-translated P points:\n";
  for (i=0; i<3; i++)
    cerr << TCp.unproject(origPtsP_[i]) << "\n";

  cerr << "\nxy-plane P points:\n";
  for (i=0; i<3; i++)
    cerr << p[i] << "\n";

  cerr << "\nRotated (by "<<theta_best<<") xy-plane Translated A points:\n";
  for (i=0; i<3; i++)
    cerr << Theta.project(a[i]) << "\n";
  
  cerr << "\nScaled (by "<<scale<<") Rotated xy-plane Translated A points:\n";
  for (i=0; i<3; i++)
    cerr << S.project(Theta.project(a[i])) << "\n";

  cerr << "\n\n\nOriginal P points:\n";
  for (i=0; i<3; i++)
    cerr << origPtsP_[i] << "\n";

  cerr << "\nCoregistered A points:\n";
  for (i=0; i<3; i++)
    cerr << transPtsA_[i] << "\n";
  
  cerr << "\nMisfit = "<<misfit<<"\n";

  cerr << "\n\n\nTCp:\n";
  TCp.print();

  cerr << "\n\n\nBpt:\n";
  Bpt.print();

  cerr << "\n\n\nS:\n";
  S.print();

  cerr << "\n\n\nTheta:\n";
  Theta.print();

  cerr << "\n\n\nBa:\n";
  Ba.print();

  cerr << "\n\n\nTC_a:\n";
  TC_a.print();

  cerr << "\n\n\nComposite Transform:\n";
  transform_.print();
#endif

  return 1;
}

}
