
/*
 *  STreeToTris: Read in a surfTree and ouput a .pts file and .tri files
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace SCICore::Containers;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;

main(int argc, char **argv) {

    SurfaceHandle handle;
    char name[100];

    if (argc !=2) {
	printf("Need the file name!\n");
	exit(0);
    }
    sprintf(name, "%s.stree", argv[1]);
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
    
    Surface *su=handle.get_rep();
    SurfTree *st=su->getSurfTree();

    if(!st){
       printf("Error getting surftree\n");
       exit(-1);
    }

    sprintf(name, "%s.pts", argv[1]);
    FILE *fout=fopen(name, "wt");
    if(!fout){
       printf("Error opening output file: %s\n", name);
       exit(-1);
    }

    int i,j;
    for (i=0; i<st->nodes.size(); i++) {
	fprintf(fout, "%lf %lf %lf\n", st->nodes[i].x(), st->nodes[i].y(),
		st->nodes[i].z());
    }
    fclose(fout);

    for (i=0; i<st->surfI.size(); i++) {
	sprintf(name, "%d.tri", i+1);
	fout=fopen(name, "wt");
	for (j=0; j<st->surfI[i].faces.size(); j++) {
	    TSElement *e=st->faces[st->surfI[i].faces[j]];
	    if (st->surfI[i].faceOrient.size()>j &&
		!st->surfI[i].faceOrient[j])
		fprintf(fout, "%d %d %d\n", e->i1+1, e->i3+1, e->i2+1);
	    else
		fprintf(fout, "%d %d %d\n", e->i1+1, e->i2+1, e->i3+1);
	}
	fclose(fout);
    }
    return 0;
}    
