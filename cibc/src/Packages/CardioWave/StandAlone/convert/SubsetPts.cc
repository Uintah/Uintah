/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SubsetPts.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Math/MusilRNG.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/TetVolField.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" ptsIn fractionOut ptsOut\n";
    return 0;
  }
  ifstream istream(argv[1]);
  double fraction = atof(argv[2]);

  BBox b;
  vector<Point> pts;
  int i, npts;
  istream >> npts;
  MusilRNG mr;
  for (i=0; i<npts; i++) {
    double x, y, z;
    istream >> x;
    if (!istream) { 
      cerr << "Error - only read "<<i<<" of "<<npts<<" points.\n"; 
      return 0; 
    }
    istream >> y >> z;
    b.extend(Point(x,y,z));
    if (mr() <= fraction) pts.push_back(Point(x,y,z));
  }

  // shuffle the points
  Point swapPt;
  int swapIdx;
  for (i=0; i<pts.size(); i++) {
    swapIdx = mr()*pts.size();
    swapPt = pts[i];
    pts[i] = pts[swapIdx];
    pts[swapIdx] = swapPt;
  }

  // jitter the points
  double n = (pow(npts, 0.3)-1);
  Vector d = b.diagonal() / (n*100);
  cerr << "jitter vector = "<<d<<"\n";
  ofstream ostream(argv[3]);
  ostream << pts.size() << "\n";
  for (i=0; i<pts.size(); i++) {
    const Point &p = pts[i];
//    Vector d0((mr()-.5)*2*d.x(), (mr()-.5)*2*d.y(), (mr()-.5)*2*d.z());
//    p += d0;
    ostream << p.x() << " " << p.y() <<" " << p.z() <<"\n";
  }
  return 0;
}
