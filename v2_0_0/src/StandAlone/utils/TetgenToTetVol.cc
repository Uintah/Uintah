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
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int readLine(FILE **f, char *buf) {
    char c;
    int cnt=0;
    do {
      if(!feof(*f) && ((c=fgetc(*f))=='#')) {
	while (!feof(*f) && ((c=fgetc(*f))!='\n'));
      }
    } while (c=='\n');
    if (feof(*f)) return 0;
    buf[cnt++]=c;
    while (!feof(*f) && ((c=fgetc(*f))!='\n')) buf[cnt++]=c;
    buf[cnt]=c;
    if (feof(*f)) return 0;
    return 1;
}

int
main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "usage: " << argv[0] << " tetgen_basename\n";
    exit(0);
  }

  TetVolMeshHandle tvm = new TetVolMesh();
  char buf[10000];
  int i, dum1, dum2, nattr, tattr, npts, ntets;
  double x, y, z;
  int i1, i2, i3, i4, c;

  Array1<int> newConds;
  char fname[1000];
  sprintf(fname, "%s.1.node", argv[1]);
  FILE *f=fopen(fname, "rt");
  if (!f) {
    cerr << "Error - failed to open "<<argv[1]<<".1.node\n";
    exit(0);
  }

  readLine(&f, buf);
  sscanf(buf, "%d %d %d %d", &npts, &dum1, &nattr, &dum2);
  cerr << "File has "<<npts<<" points and "<<nattr<<" attributes.\n";
  if (nattr != 0) {
    cerr << "Error - only know how to handle zero node attributes.\n";
    exit(0);
  }

  Array1<Point> allPts;
  
  // idx x y z
  for (i=0; i<npts; i++) {
    readLine(&f, buf);
    sscanf(buf, "%d %lf %lf %lf", &dum1, &x, &y, &z);
    allPts.add(Point(x,y,z));
    tvm->add_point(Point(x,y,z));
  }
  fclose(f);
  //  tvm->compute_nodes();

  sprintf(fname, "%s.1.ele", argv[1]);
  f=fopen(fname, "rt");
  if (!f) {
    cerr << "Error - failed to open "<<argv[1]<<".1.ele\n";
    exit(0);
  }

  readLine(&f, buf);
  sscanf(buf, "%d %d %d", &ntets, &dum1, &tattr);
  cerr << "File has "<<ntets<<" tets and "<<tattr<<" attributes.\n";
  if (tattr != 1) {
    cerr << "Error - only know how to handle one tet attribute.\n";
    exit(0);
  }

  Array1<int> fdata;
  // idx i1 i2 i3 i4 cond
  for (i=0; i<ntets; i++) {
    readLine(&f, buf);
    sscanf(buf, "%d %d %d %d %d %d", &dum1, &i1, &i2, &i3, &i4, &c);
    tvm->add_tet(i1-1, i2-1, i3-1, i4-1);
    fdata.add(c);
  }
  fclose(f);

  TetVolField<int> *tvi = scinew TetVolField<int>(tvm, Field::CELL);
  vector<pair<string, Tensor> > tens;
  tens.resize(1);
  tens[0] = pair<string, Tensor>("inside", Tensor(1.0));
  tvi->set_property("conductivity_table", tens, false);
  for (i=0; i<ntets; i++)
    tvi->fdata()[i] = fdata[i];

  FieldHandle tvH(tvi);
  
  sprintf(fname, "%s.tvt.fld", argv[1]);

  TextPiostream out_stream(fname, Piostream::Write);
  Pio(out_stream, tvH);

  return 0;  
}    
