
/*
 *  MeshToSFUG: Read in a Mesh output a SFUG (values @ node = y coord)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
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

main(int argc, char **argv) {

    int i, j;

    if (argc !=3) {
	printf("Need the infile and outfile names.\n");
	exit(0);
    }

    Piostream* stream=auto_istream(argv[1]);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", argv[1]);
	exit(0);
    }

    ScalarFieldHandle sfH;
    Pio(*stream, sfH);
    if (!sfH.get_rep()) {
	printf("Error reading SFUG from file %s.  Exiting...\n", argv[1]);
	exit(0);
    }

    ScalarFieldUG *sfIn = dynamic_cast<ScalarFieldUG*>(sfH.get_rep());
    if (!sfIn) {
	printf("Error - input ScalarField %s wasn't an SFUG.  Exiting...\n", argv[1]);
	exit(0);
    }
    ScalarFieldUG *sfOut;

    sfIn->mesh->compute_neighbors();
    sfIn->mesh->compute_face_neighbors();

    if (sfIn->typ == ScalarFieldUG::NodalValues) {
	cerr << "SFUG had nodal values - converting to element values.\n";
	sfOut=scinew ScalarFieldUG(sfIn->mesh, ScalarFieldUG::ElementValues);
	for (i=0; i<sfOut->data.size(); i++) {
	    sfOut->data[i]=0;
	    Element *e=sfIn->mesh->elems[i];
	    for (j=0; j<4; j++) sfOut->data[i] += sfIn->data[e->n[j]] / 4.;
	}
    } else {
	cerr << "SFUG had element values - converting to nodal values.\n";
	sfOut=scinew ScalarFieldUG(sfIn->mesh, ScalarFieldUG::NodalValues);
	for (i=0; i<sfOut->data.size(); i++) {
	    sfOut->data[i]=0;
	    NodeHandle n(sfIn->mesh->nodes[i]);
	    for (j=0; j<n->elems.size(); j++) 
		sfOut->data[i] += sfIn->data[n->elems[j]] / n->elems.size();
	}
    }

    BinaryPiostream stream2(argv[2], Piostream::Write);
    ScalarFieldHandle sfH2(sfOut);
    Pio(stream2, sfH2);
    sfIn->mesh=0;	// so we don't crash on exit
    return 0;
}
