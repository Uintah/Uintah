
/*
 *  MakeRadialCylinder: Creates a mesh for a cylinder using radial symmetry
 *
 *
 *  Written by:
 *   Kris Zyp
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Trig.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

// push all of the nodes into concentric circles
// allows angles to increment and wrap around when necessary
int wrapAroundAngle(int angle, int limit){  
  if ((angle) == limit) return 0;
  else return angle;
}
     
Array1<int> genPtsAndTets(int nr, int nz, TetVolMesh *mesh) {
  Array1<float> zPos;
  Array1<float> rPos;
  int currIdx,parity;
  int r1,r2,angleInner,angleOuter,z;

  if (nr==0 && nz==0) {
    zPos.resize(22);
    rPos.resize(14);
    nz=22;
    zPos[0] = 0;
    zPos[1] =9.5;
    zPos[2] =19.05;
    zPos[3] =29.36;
    zPos[4] =39.6875;
    zPos[5] =50;
    zPos[6] =60.325;
    zPos[7] =70;
    zPos[8] =79.375;
    zPos[9] =89.7;
    zPos[10] =100.0125;
    zPos[11] =109.53;
    zPos[12] =119.0625;
    zPos[13] =129.28;
    zPos[14] =139.7;
    zPos[15] =150.02;
    zPos[16] =160.3375;
    zPos[17] =169.85;
    zPos[18] =179.3875;
    zPos[19] =189.7;
    zPos[20] =200.025;
    zPos[21] =204.7875;

    nr=14;
    rPos[0] = 0;
    rPos[1] = 7.62;
    rPos[2] = 7.62*2;
    rPos[3] = 7.62*3;
    rPos[4] = 7.62*4;
    rPos[5] = 7.62*5;
    rPos[6] = 7.62*6;
    rPos[7] = 7.62* 7;
    rPos[8] = 7.62* 8;
    rPos[9] = 7.62*9;
    rPos[10] = 76.2;
    rPos[11] = 82.55;
    rPos[12] = 89.65;
    rPos[13] = 96.75;
  } else {
    rPos.resize(nr);
    zPos.resize(nz);
    for (int z=0; z<nz; z++) zPos[z]=z/(nz-1.)*204.7875;
    for (int r=0; r<nr; r++) rPos[r]=r/(nr-1.)*96.75;
  }
  Array1<int> conds;
  Array3<int> nodes(nr, nr * 6, nz);
  Array1<int> c(6);
  r1 = 0;
  parity =0;
  currIdx =0;
  for (z=0; z<nz; z++) {
    mesh->add_point(Point(0,0,zPos[z]));  // make center point
    nodes(0,0,z) = currIdx++;
    for (r2=1; r2<nr; r2++)
      for (angleOuter=0; angleOuter<r2*6; angleOuter++) {
	mesh->add_point(Point(rPos[r2]*cos(angleOuter * PI/3 / r2),
			      rPos[r2]*sin(angleOuter * PI/3 / r2),
			      zPos[z])); // create radial points
	nodes(r2,angleOuter,z) = currIdx++;
      }
  }
  for (z=0; z<nz-1; z++)
    for (r1=0; r1<nr-1; r1++) {
      r2 = r1 + 1;
      angleOuter = 0;
      angleInner = 0;
      while (angleOuter < 6 * r2) {
	parity++;
	// move along both radii at the same rate, this is a normal pie slice
	if (angleOuter * r1 <= angleInner * r2) {  
	  // the corners of the pie
	  c[0] = nodes(r1,wrapAroundAngle(angleInner,6*r1),z);  
	  c[1] = nodes(r1,wrapAroundAngle(angleInner,6*r1),z+1);
	  c[2] = nodes(r2,angleOuter,z);
	  c[3] = nodes(r2,angleOuter,z+1);
	  c[4] = nodes(r2,wrapAroundAngle(angleOuter+1,6*r2),z);
	  c[5] = nodes(r2,wrapAroundAngle(angleOuter+1,6*r2),z+1);
	  angleOuter++;
	} else {  // this one is the backwards pie slice (pointing outwards)
	  c[0] = nodes(r2,angleOuter,z);
	  c[1] = nodes(r2,angleOuter,z+1);
	  c[2] = nodes(r1,wrapAroundAngle(angleInner+1,6*r1),z);
	  c[3] = nodes(r1,wrapAroundAngle(angleInner+1,6*r1),z+1);
	  c[4] = nodes(r1,angleInner,z);
	  c[5] = nodes(r1,angleInner,z+1);
	  angleInner++;
	}
	if (parity%2) {  // use parity to keep switching direction
	  mesh->add_tet(c[0],c[2],c[3],c[4]);  // slice the pie up into tets
	  mesh->add_tet(c[1],c[3],c[4],c[5]);
	  mesh->add_tet(c[0],c[1],c[3],c[4]);
	} else {
	  mesh->add_tet(c[1],c[2],c[3],c[5]);
	  mesh->add_tet(c[0],c[2],c[4],c[5]);
	  mesh->add_tet(c[0],c[1],c[2],c[5]);
	}
	if (r1 < (nr/14.*9.5)) { // inner region conductivity index is 0
	  conds.add(0); conds.add(0); conds.add(0);
	} else if (r1 < 1+(nr/14.*9.5)) { // middle region cond index is 1
	  conds.add(1); conds.add(1); conds.add(1);
	} else { // outter region conductivity index is 2
	  conds.add(2); conds.add(2); conds.add(2);	
	}
      }
    }
  return conds;
}

int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 4) {
    cerr << "Usage: MakeRadialCylinder [nr nz] outputfield\n";
    exit(0);
  }
  TetVolMesh *mesh = scinew TetVolMesh;
  int i;
  int nr, nz;
  nr=nz=0;
  char *outname;
  if (argc == 2) {
    outname = argv[1];
  } else {
    nr = atoi(argv[1]);
    nz = atoi(argv[2]);
    outname = argv[3];
  }

//  Array1<int> data = genPtsAndTets(14, 22, mesh);  // generate everything
  Array1<int> data = genPtsAndTets(nr, nz, mesh);  // generate everything

  vector<pair<string, Tensor> > conds(3);
  conds[0].first = "0 conductivity";
  conds[0].second = Tensor(0.3012);
  conds[1].first = "1 conductivity";
  conds[1].second = Tensor(0.1506);
  conds[2].first = "2 conductivity";
  conds[2].second = Tensor(0.4518);

  TetVolMeshHandle tvmH(mesh);
  TetVolField<int> *tv = scinew TetVolField<int>(tvmH, Field::CELL);

  for (i=0; i<data.size(); i++)
    tv->fdata()[i]=data[i];
  FieldHandle mesh_out_(tv);

  mesh_out_->store("data_storage", string("table"), false);
  mesh_out_->store("name", string("conductivity"), false);
  mesh_out_->store("conductivity_table", conds, false);
  Piostream* stream = scinew TextPiostream(outname, Piostream::Write);
  Pio(*stream, mesh_out_);
  delete(stream);
  return 1;
}
