
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

#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

main(int argc, char **argv) {

    MeshHandle handle;
    char name[100];
    char *oname=argv[1];


    if (argc !=2) {
	printf("Need the file name!\n");
	exit(0);
    }
    sprintf(name, "%s.mesh", oname);
    Piostream* stream=auto_istream(name);
    if (!stream) {
	printf("Couldn't open file %s.  Exiting...\n", name);
	exit(0);
    }
    Pio(*stream, handle);
    if (!handle.get_rep()) {
	printf("Error reading Mesh from file %s.  Exiting...\n", name);
	exit(0);
    }

    Mesh *m=handle.get_rep();

    sprintf(name, "%s.pts", oname);
    FILE *fout=fopen(name, "wt");
    fprintf(fout, "%d\n", m->nodes.size());
    int i;
    for (i=0; i<m->nodes.size(); i++) {
	fprintf(fout, "%lf %lf %lf\n", m->nodes[i].p.x(), m->nodes[i].p.y(),
		m->nodes[i].p.z());
    }
    fclose(fout);

    sprintf(name, "%s.tetras", oname);
    fout=fopen(name, "wt");
    fprintf(fout, "%d\n", m->elems.size());
    for (i=0; i<m->elems.size(); i++) {
//	fprintf(fout, "%d %d %d %d\n", m->elems[i]->n[0]+1,
//		m->elems[i]->n[1]+1, m->elems[i]->n[2]+1,
//		m->elems[i]->n[3]+1);
	fprintf(fout, "%d %d %d %d %d\n", m->elems[i]->n[0]+1,
		m->elems[i]->n[1]+1, m->elems[i]->n[2]+1,
		m->elems[i]->n[3]+1, m->elems[i]->cond);
    }
    fclose(fout);

    sprintf(name, "%s.conds", oname);
    fout=fopen(name, "wt");
    fprintf(fout, "%d\n", m->cond_tensors.size());
    for (i=0; i<m->cond_tensors.size(); i++) {
	fprintf(fout, "%lf %lf %lf %lf %lf %lf\n",
		m->cond_tensors[i][0],
		m->cond_tensors[i][1],
		m->cond_tensors[i][2],
		m->cond_tensors[i][3],
		m->cond_tensors[i][4],
		m->cond_tensors[i][5]);
    }
    fclose(fout);
    return 0;
}    
