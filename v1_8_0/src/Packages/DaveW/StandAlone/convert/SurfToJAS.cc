
/*
 *  SurfToJAS: Read in a surface, and output a .tri and .pts file
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
#include <Core/Datatypes/TriSurfFieldace.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using namespace SCIRun;

main(int argc, char **argv) {

    SurfaceHandle handle;
    char name[100];

    if (argc !=2) {
	printf("Need the file name!\n");
	exit(0);
    }
    sprintf(name, "%s.surf", argv[1]);
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
    TriSurfFieldace *ts=su->getTriSurfFieldace();
    if(!ts){
       printf("Error getting ts\n");
       exit(-1);
    }

    sprintf(name, "%s.pts", argv[1]);
    FILE *fout=fopen(name, "wt");
    if(!fout){
       printf("Error opening output file: %s\n", name);
       exit(-1);
    }
    fprintf(fout, "%d\n", ts->points.size());
//    fprintf(stderr, "%d\n", ts->points.size());
    int i;
    for (i=0; i<ts->points.size(); i++) {
	fprintf(fout, "%lf %lf %lf\n", ts->points[i].x(), ts->points[i].y(),
		ts->points[i].z());
//	fprintf(stderr, "%lf %lf %lf\n", ts->points[i].x(), ts->points[i].y(),
//		ts->points[i].z());
    }
    fclose(fout);

    sprintf(name, "%s.tri", argv[1]);
    fout=fopen(name, "wt");
    for (i=0; i<ts->elements.size(); i++) {
	fprintf(fout, "%d %d %d\n", ts->elements[i]->i1+1,
		ts->elements[i]->i2+1, ts->elements[i]->i3+1);
    }
    fclose(fout);

    if (ts->normType != TriSurfFieldace::NrmlsNone) {
	sprintf(name, "%s.nrm", argv[1]);
	fout=fopen(name, "wt");
	fprintf(fout, "%d\n", ts->normals.size());
	for (i=0; i<ts->normals.size(); i++) {
	    fprintf(fout, "%lf %lf %lf\n", ts->normals[i].x(), 
		    ts->normals[i].y(), ts->normals[i].z());
	}
	fclose(fout);
    }
	
    return 0;
}    
