/*
 *  EGItoMat.cc: Read in the EGI ampm file and output two matrices: 
 *                  one with a matrix of electrode voltages, one with
 *                  a permutation matrix for throwing out the bad electrodes
 *                  from the first vector
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   November 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
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
    int selIdx=-1;
    if (argc != 5 && argc != 4) {
	printf("argc=%d\n", argc);
	printf("Usage: %s EGIfile index(sel) sel.out [volts.out]\n", argv[0]);
	exit(0);
    }
    selIdx=atoi(argv[2]);
    FILE *fin = fopen(argv[1], "rt");
    if (!fin) {
      printf("Error - unable to open input file %s\n", argv[1]);
      exit(0);
    }
    int nelecs, npairs, refIdx;
    double amps, rad1, rad2, rad3, height;

    if (fscanf(fin, "%d %d %lf %d\n", &nelecs, &npairs, &amps, &refIdx) != 4) {
      printf("Error reading input file.\n");
      exit(0);
    }
    if (selIdx<0 || selIdx>=npairs) {
      printf("Error - only %d pairs specified, can't select pair %d for selection matrix.\n", npairs, selIdx);
      exit(0);
    }
    if (fscanf(fin, "%lf %lf %lf %lf", &rad1, &rad2, &rad3, &height) != 4) {
      printf("Error reading input file\n");
      exit(0);
    }
    printf("Selecting pair %d\n", selIdx);

    DenseMatrix *elecVolt = new DenseMatrix(npairs, nelecs);
    Array1<int> goodElecs;
    int i, j;
    int srcIdx, sinkIdx, badVal, elecIdx;
    double val;
    for (i=0; i<npairs; i++) for (j=0; j<nelecs; j++) (*elecVolt)[i][j]=-1234;
    for (i=0; i<npairs; i++) {
      if (fscanf(fin, "%d %d", &srcIdx, &sinkIdx) != 2) {
	printf("Not enough pairs...\n");
	exit(0);
      }
      if (i==selIdx)
	printf("srcIdx=%d  sinkIdx=%d\n", srcIdx, sinkIdx);
      for (j=0; j<nelecs; j++) {
	if (fscanf(fin, "%d %lf %d", &elecIdx, &val, &badVal) != 3) {
	  printf("Not enough electrodes...\n");
	  exit(0);
	}
	if (i==selIdx){ // store the bad-index (permutation) matrix
	  if (!badVal)
	    goodElecs.add(j);
	}
	(*elecVolt)[i][j]=val;
      }
    }
    
    DenseMatrix *selectionMat = new DenseMatrix(goodElecs.size(),nelecs);
    for (i=0; i<goodElecs.size(); i++) 
      for (j=0; j<nelecs; j++) 
	(*selectionMat)[i][j]=0;

    for (i=0; i<goodElecs.size(); i++) 
      (*selectionMat)[i][goodElecs[i]]=1;

    TextPiostream sel(argv[3], Piostream::Write);
    MatrixHandle selectionMatH=selectionMat;
    Pio(sel, selectionMatH);
    if (argc == 5) {
      TextPiostream volts(argv[4], Piostream::Write);
      MatrixHandle elecVoltH=elecVolt;
      Pio(volts, elecVoltH);
    }
}
