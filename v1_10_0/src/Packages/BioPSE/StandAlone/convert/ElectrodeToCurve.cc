/*
 *  ElectrodeToContour: Read in an electrode and save it out as a contour
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/CurveField.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s ElectrodeToContour electrode CurveField\n", argv[0]);
    exit(0);
  }
  FILE *fin = fopen(argv[1], "rt");
  if (!fin) {
    printf("Error - unable to open input file %s\n", argv[1]);
    exit(0);
  }
  int nnodes;
  int ninner;
  
  if (fscanf(fin, "%d %d\n", &ninner, &nnodes) != 2) {
    printf("Error reading input file.\n");
    exit(0);
  }
  
  CurveMeshHandle cm(scinew CurveMesh);
  double x,y;
  if (fscanf(fin, "%lf %lf\n", &x, &y) != 2) {
    printf("Error reading input file.\n");
    exit(0);
  }
  cm->add_node(Point(x,y,0)); // handle
  cm->add_node(Point(x,y,0.10)); // handle
  cm->add_node(Point(x+0.10,y,0.10)); // handle
  cm->add_node(Point(x-0.10,y,0.10)); // handle
  cm->add_node(Point(x,y+0.10,0.10)); // handle
  cm->add_node(Point(x,y-0.10,0.10)); // handle
  cm->add_edge(0,1);
  cm->add_edge(2,3);
  cm->add_edge(4,5);

  if (fscanf(fin, "%lf %lf\n", &x, &y) != 2) {
    printf("Error reading input file.\n");
    exit(0);
  }
  cm->add_node(Point(x,y,0));

  int i;
  for (i=7; i<nnodes+6; i++) {
    if (fscanf(fin, "%lf %lf\n", &x, &y) != 2) {
      printf("Error reading input file.\n");
      exit(0);
    }
    cm->add_node(Point(x,y,0));
    if (i!=(ninner+6)) cm->add_edge(i-1,i);
    else cm->add_edge(i-1,6);
  }
  cm->add_edge(i-1,(ninner+6));

  CurveField<double> *cf = scinew CurveField<double>(cm, Field::NODE);
  for (i=0; i<(nnodes+6); i++) {
    if (i<6) {
      cf->fdata()[i]=1;
    } else if (i<(ninner+6)) {
      cf->fdata()[i]=0;
    } else {
      cf->fdata()[i]=2;
    }
  }
  
  FieldHandle fld(cf);
  TextPiostream stream(argv[2], Piostream::Write);
  Pio(stream, fld);
}
