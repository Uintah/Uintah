
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

    int i;
    MeshHandle mesh;

    if (argc !=2) {
	printf("Need the basename\n");
	exit(0);
    }
    clString fin(clString(argv[1])+clString(".mesh"));
    clString fout(clString(argv[1])+clString(".sfug"));

    Piostream* stream=auto_istream(fin);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", fin());
	exit(0);
    }
    Pio(*stream, mesh);
    if (!mesh.get_rep()) {
	printf("Error reading Mesh from file %s.  Exiting...\n", fin());
	exit(0);
    }

    ScalarFieldUG *sfug=scinew ScalarFieldUG(mesh, ScalarFieldUG::NodalValues);
    
    for (i=0; i<sfug->data.size(); i++)
	sfug->data[i] = mesh->nodes[i]->p.y();

    BinaryPiostream stream2(fout, Piostream::Write);
    ScalarFieldHandle sfH(sfug);
    Pio(stream2, sfH);
    return 0;
}    
