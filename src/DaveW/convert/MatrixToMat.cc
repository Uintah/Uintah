
/*
 *  MatrixToMat: Read in a SCIRun matrix, and output a .mat file for my
 *		 'solve' inverse solving program.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Datatypes/Matrix.h>
#include <SCICore/Geometry/Point.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCICore::Containers;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;

int main(int argc, char **argv) {

    MatrixHandle handle;
    char name[100];

    if (argc !=2) {
	printf("%s fname\n", argv[0]);
	exit(0);
    }
    sprintf(name, "%s.matrix", argv[1]);
    Piostream* stream=auto_istream(name);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", name);
	exit(0);
    }
    Pio(*stream, handle);
    if (!handle.get_rep()) {
	printf("Error reading surface from file %s.  Exiting...\n", name);
	exit(0);
    }

    Matrix *m=handle.get_rep();
    int nr=m->nrows();
    int nc=m->ncols();
    int nnz=0;
    Array1<int> idx;
    Array1<double> v;
    int i;
    for (i=0; i<nr; i++) {
	m->getRowNonzeros(i, idx, v);
	nnz+=idx.size();
    }
    sprintf(name, "%s.mat", argv[1]);
    FILE *f=fopen(name, "wt");
    fprintf(f, "%d %d %d\n", nr, nc, nnz);

    for (i=0; i<nr; i++) {
	m->getRowNonzeros(i, idx, v);
	for (int j=0; j<idx.size(); j++) {
	    fprintf(f, "%d %d %lf\n", i, idx[j], v[j]);
	}	 
    }
    fclose(f);
    return 1;
}    
