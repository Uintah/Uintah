/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distetbuted under the License is distetbuted on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  FlatToTetVolField.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TetVolField.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Array1.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int readLine(FILE **f, char *buf) {
    char c=0;
    int cnt=0;
    while (!feof(*f) && ((c=fgetc(*f))!='\n')) buf[cnt++]=c;
    buf[cnt]=c;
    if (feof(*f)) return 0;
    return 1;
}

int
main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "usage: " << argv[0] << " vdt_t3d_file scirun_tvt\n";
    exit(0);
  }

  TetVolMeshHandle tvm = new TetVolMesh();
  char buf[10000];
  int i, dum1, dum2, npts, ntets;
  double x, y, z;
  int i1, i2, i3, i4, c;

  vector<int> newConds;
  FILE *f=fopen(argv[1], "rt");
  if (!f) {
    cerr << "Error - failed to open "<<argv[1]<<".t3d\n";
    exit(0);
  }

  // ! VDT (C) 1998 Petr Krysl
  readLine(&f, buf);
  
  // nnodes nedges ntriangles ntetras
  readLine(&f, buf);
  sscanf(buf, "%d %d %d %d", &npts, &dum1, &dum2, &ntets);
  cerr << "File has "<<npts<<" points and "<<ntets<<" tets.\n";
  
  vector<Point> allPts;
  vector<int> ptsMap(npts*4, -1);

  // etype edegree
  readLine(&f, buf);
  
  // ! points
  readLine(&f, buf);
  
  // idx x y z
  for (i=0; i<npts; i++) {
    readLine(&f, buf);
    sscanf(buf, "%d %lf %lf %lf", &dum1, &x, &y, &z);
    ptsMap[dum1] = allPts.size();
    allPts.push_back(Point(x,y,z));
    tvm->add_point(Point(x,y,z));
  }
  
  // ! tets
  readLine(&f, buf);

  int maxCond = 0;
  Array1<int> conds;

  // idx i1 i2 i3 i4 IGNORE cond
  for (i=0; i<ntets; i++) {
    readLine(&f, buf);
    sscanf(buf, "%d %d %d %d %d %d %d", &dum1, &i1, &i2, &i3, &i4, &dum2,
	   &c);
    if (c > maxCond) maxCond=c;
    tvm->add_tet(ptsMap[i1], ptsMap[i2], ptsMap[i3], ptsMap[i4]);
    conds.add(c);
  }
  
  TetVolField<int> *tvi = scinew TetVolField<int>(tvm, Field::CELL);
  vector<pair<string, Tensor> > tens;
  tens.resize(maxCond);
  for (i=0; i<maxCond; i++)
    tens[i] = pair<string, Tensor>(string("matl")+to_string(i), Tensor(1.0));
  tvi->set_property("conductivity_table", tens, false);
  for (i=0; i<ntets; i++)
    tvi->fdata()[i] = conds[i]-1;

  FieldHandle tvH(tvi);
  TextPiostream out_stream(argv[2], Piostream::Write);
  Pio(out_stream, tvH);
  return 0;  
}    
