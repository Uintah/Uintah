
/*
 *  SurfToVDT: Read in a surface, and output a .vin file
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
using namespace DaveW::Datatypes;
using std::cerr;

#define SWAP(a,b) {int dum=(a);(a)=(b);(b)=dum;}

main(int argc, char **argv) {

    SurfaceHandle handle;
    char name[100];

    if (argc !=2) {
	printf("Need the file name!\n");
	exit(0);
    }

    Piostream* stream=auto_istream(clString(clString(argv[1])+".st"));
    if (!stream) stream=auto_istream(clString(clString(argv[1])+".stree"));
    if (!stream) {
	printf("Couldn't open file %s.st of %s.stree.  Exiting...\n", argv[1], argv[1]);
	exit(0);
    }
    Pio(*stream, handle);
    if (!handle.get_rep()) {
	printf("Error reading surface from file %s.  Exiting...\n", name);
	exit(0);
    }
    Surface *su=handle.get_rep();
    SurfTree* st=dynamic_cast<SurfTree*>(su);
    if(!st){
       printf("Error getting SurfTree\n");
       exit(-1);
    }

    sprintf(name, "%s.vin", argv[1]);
    FILE *fout=fopen(name, "wt");
    if(!fout){
       printf("Error opening output file: %s\n", name);
       exit(-1);
    }
    fprintf(fout, "input_version \"$Id$\"\n", name);
    fprintf(fout, "mesh_size %lf\n", (st->nodes[0]-st->nodes[1]).length());
    fprintf(fout, "gen_interior_pts\n");

    int i;
    for (i=0; i<st->nodes.size(); i++)
	fprintf(fout, "pt %d %lf %lf %lf\n", i+1, st->nodes[i].x(), st->nodes[i].y(), st->nodes[i].z());
    fprintf(fout, "! boundary facets\n");
    for (i=0; i<st->faceI.size(); i++) {
	int i1, i2, i3;
	i1=st->faces[i]->i1+1;
	i2=st->faces[i]->i2+1;
	i3=st->faces[i]->i3+1;
	int r1, r2;
	r1=st->faceI[i].surfIdx[0];
	if (st->faceI[i].surfIdx.size() < 2)
	    cerr << "Error - face "<<i<<" only has one region??\n";
	r2=st->faceI[i].surfIdx[1];
        int p1=st->faceI[i].patchIdx+1;
	p1=1;
//	cerr << "i1="<<i1<<"  i2="<<i2<<"  i3="<<i3<<"  r1="<<r1<<"  r2="<<r2<<"  p1="<<p1<<"  so="<<st->faceI[i].surfOrient[0]<<"\n";
	if (!st->faceI[i].surfOrient[0]) SWAP(i1,i2)
        if (r1<r2) { SWAP(i1,i2) SWAP(r1,r2) }
	fprintf(fout, "bsf %d %d %d  %d %d", i1, i2, i3, p1, r1);
	if (r2) fprintf(fout, " %d", r2);
	fprintf(fout, "\n");
    }
    fclose(fout);
    return 0;
}    
