
/*
 *  DuckReader: Reads data from the cylinder electrode EEG analysis
 *	done at the University of Oregon
 *
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
    //    if (argc != 3) {
    //	cerr << "Usage: MakeRadialCylinder radius height\n";
    //	exit(0);
    //}
    int electrodes, tests;
    //nr=atoi(argv[1]);  // get radius
    //nz=atoi(argv[2]);  // get height
    int i,j;
    int tmpi;
    float tmpf;
    char* inputFileName = argv[argc-1];
    char* outputFileName;
    ifstream in(inputFileName);
    if (!in) {
	cerr << "file can't be opened\n";
	return 1;
    }
    cerr << "starting to read\n";
    in >> electrodes;
    in >> tests;
    ColumnMatrix *values = scinew ColumnMatrix(electrodes);
    ColumnMatrix *electrodeSelection = scinew ColumnMatrix(2);

    DenseMatrix *integrity = scinew DenseMatrix(electrodes, electrodes);
    cerr << "electrodes " << electrodes << "  tests " << tests << endl;

    for(i=0;i<6;i++) {
	in >> tmpf;
    }    
    cerr << "reading main data\n";
   for(i=0;i<tests;i++) {

       in >> tmpi;
       (*electrodeSelection)[0] = tmpi;
       in >> tmpi;
       (*electrodeSelection)[1] = tmpi;

	for (j=0;j<electrodes;j++) {
	    in >> tmpi;
	    if (tmpi != j + 1) {
		cerr << "inconsistent table of values.\n";
		return 1;
	    }
	    in >> tmpf;
	    (*values)[j] = tmpf;
	    in >> tmpi;
	    (*integrity)[j][j] = (tmpi == 0);
	}
   }
   cerr << "saving data\n";
    clString fname(clString("/home/cs/zyp/PSE/src/Packages/DaveW/convert/integrity.1"));
    Piostream* stream = scinew TextPiostream(fname, Piostream::Write);
    MatrixHandle integrityHandle(integrity);  // save it all
    Pio(*stream, integrityHandle);
    delete(stream);

    clString zcmfname(clString("/home/cs/zyp/PSE/src/Packages/DaveW/convert/values.1"));
    stream = scinew TextPiostream(zcmfname, Piostream::Write);  // save the cylinder height values
    ColumnMatrixHandle vcMH(values);  // save it all
    Pio(*stream, vcMH);
    delete(stream);

    clString ccmfname(clString("/home/cs/zyp/PSE/src/Packages/DaveW/convert/es.1"));
    stream = scinew TextPiostream(ccmfname, Piostream::Write); 
    ColumnMatrixHandle escMH(electrodeSelection);  // save the cylinder conductivity values
    Pio(*stream, escMH);
    delete(stream);

}
