
/*
 *  DuckReader: Writes data from a forward problem solve into the format used
 *     at the University of Oregon
 *  usage:
 *  DuckReader outputfilename columnmatrix_file_of_values columnmatrix_file_of_injection_pair
 *  Written by:
 *   Kris Zyp
 *   Department of Computer Science
 *   University of Utah
 *   October 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/ColumnMatrix.h>
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

int main(int argc, char *argv[]) {
    int electrodes, tests;
    //nr=atoi(argv[1]);  // get radius
    //nz=atoi(argv[2]);  // get height
    int i;
    int tmpi;
   FILE *txtfile;
    txtfile = fopen(argv[1],"w");            // open file, overwrite data
    ColumnMatrixHandle cmh;
    if (argc !=4) {
	printf("Need the input values and injections and output filenames\n");
	exit(0);
    }

    Piostream* stream=auto_istream(argv[2]);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", argv[2]);
	exit(0);
    }
    Pio(*stream, cmh);
    if (!cmh.get_rep()) {
	printf("Error reading ColumnMatrix from file %s.  Exiting...\n", argv[2]);
	exit(0);
    }
    ColumnMatrix* values = cmh.get_rep();
    delete(stream);
    stream=auto_istream(argv[3]);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", argv[3]);
	exit(0);
    }
    Pio(*stream, cmh);
    if (!cmh.get_rep()) {
	printf("Error reading ColumnMatrix from file %s.  Exiting...\n", argv[3]);
	exit(0);
    }
    ColumnMatrix* injections = cmh.get_rep();
    delete(stream);
    electrodes = values->nrows();
    tests = 1;
    cerr << "starting to write\n";
    fprintf(txtfile,"%14d %11d      15.8 %11d    \n   7.6        8.3       9.8       20.6\n",electrodes,tests,electrodes);
    fprintf(txtfile,"%13.0f       %8.0f\n",(*injections)[0],(*injections)[1]);
    for (i=0;i<electrodes;i++)
	fprintf(txtfile,"%13d   %8.4f          0\n",i+1,(*values)[i]);
    fclose(txtfile);
}
