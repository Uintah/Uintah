
/*
 *  MeshToJAS: Read in a Mesh, and output a .tetras, .pts and .conds files
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
#include <SCICore/Containers/Queue.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
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

    MeshHandle mesh;
    int i, j, k;
    if (argc !=4) {
	printf("Need the old file name and new file names!\n");
	exit(0);
    }
    Piostream* stream=auto_istream(argv[1]);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", argv[1]);
	exit(0);
    }
    Pio(*stream, mesh);
    if (!mesh.get_rep()) {
	printf("Error reading Mesh from file %s.  Exiting...\n", argv[1]);
	exit(0);
    }

    Array1<int> visited(mesh->nodes.size());
    visited.initialize(0);
    Queue<int> q;
    q.append(0);
    visited[0]=1;
    while (!q.is_empty()) {
	int n=q.pop();
	for (j=0; j<mesh->nodes[n]->elems.size(); j++) {
	    Element *e=mesh->elems[mesh->nodes[n]->elems[j]];
	    for (k=0; k<4; k++) {
		int nn=e->n[k];
		if (!visited[nn]) {
		    visited[nn]=1;
		    q.append(nn);
		}
	    }
	}
    }

    // starting with node 0, visit as many nodes as we can...

    int count=0;
    for (i=0; i<visited.size(); i++) {
	count+=visited[i];
	if (!visited[i]) mesh->nodes[i]=0;
    }
    cerr << "Visited "<<count<<" out of "<<visited.size()<<" nodes.\n";

    cerr << "mesh->nodes.size() == "<<mesh->nodes.size()<<"  mesh->elems.size()="<<mesh->elems.size()<<"\n";

    mesh->pack_all();

    cerr << "mesh->nodes.size() == "<<mesh->nodes.size()<<"  mesh->elems.size()="<<mesh->elems.size()<<"\n";

#if 0
    Array1<Element *> elems=mesh->elems;
    Array1<Element *> badElems;
    mesh->elems.resize(0);
    for (i=0; i<elems.size(); i++) {
	int ok=1;
	for (j=0; j<4; j++) if (!visited[elems[i]->n[j]]) ok=0;
	if (ok) mesh->elems.add(elems[i]);
	else badElems.add(elems[i]);
    }

#endif
    BinaryPiostream stream2(clString(argv[2]), Piostream::Write);
    Pio(stream2, mesh);
#if 0
    mesh->elems=badElems;
    BinaryPiostream stream3(clString(argv[3]), Piostream::Write);
    Pio(stream3, mesh);
#endif
    return 0;
}    
