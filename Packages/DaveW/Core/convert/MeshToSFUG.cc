
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

#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int main(int argc, char **argv) {

    int i;
    MeshHandle mesh;

    if (argc !=2) {
	printf("Need the basename\n");
	exit(0);
    }
    clString base1(argv[1]);
    clString fin(base1+".mesh");
    clString fout(base1+".sfug");

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
