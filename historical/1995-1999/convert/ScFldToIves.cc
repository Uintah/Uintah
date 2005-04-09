
/*
 *  ScFldToIves: Read in a scalar field, and output a .dat file for ives
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Classlib/Array1.h>
#include <stdio.h>

main(int argc, char **argv) {

    ScalarFieldHandle handle;
    char name[100];

    if (argc !=2) {
	printf("Need the file name!\n");
	exit(0);
    }
    Piostream* stream=auto_istream(argv[1]);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", name);
	exit(0);
    }
    Pio(*stream, handle);
    if (!handle.get_rep()) {
	printf("Error reading surface from file %s.  Exiting...\n", name);
	exit(0);
    }
    ScalarField *sf=handle.get_rep();
    ScalarFieldRG *sfrg=sf->getRG();

    sprintf(name, "%s.dat", argv[1]);
    ofstream fout(name);
	int max=0;
    for (int i=0; i<sfrg->nx; i++)
	for (int j=0; j<sfrg->ny; j++)
		for (int k=0; k<sfrg->nz; k++) {
			char c=(char)sfrg->grid(i,j,k);
			if (c>max)max=c;
			fout << c;
		}
     cout << "Max value from Scalar field was: " << max << "\n";
}    
