
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
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#define AIR_CONDUCTIVITY 0.0
#define SKIN_CONDUCTIVITY 1.0
#define BONE_CONDUCTIVITY 0.05
#define CSF_CONDUCTIVITY 4.620
#define GREY_CONDUCTIVITY 1.0
#define WHITE_CONDUCTIVITY 0.43

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCICore::Containers;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;

int main(int argc, char **argv) {

    int i,j;
    MeshHandle mesh;

    if (argc !=3) {
	printf("Need the old file name and new file name!\n");
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

    // fix stuff here
    
    // change some of the conductivities
    cerr << "Before changing/sorting...\n";
    for (i=0; i<30; i++) {
	int first=1;
	int min=0;
	int max=0;
	int count=0;
	for (j=0; j<mesh->elems.size(); j++) {
	    if (mesh->elems[j]->cond == i) {
		if (first) { min=j; first=0; }
		max=j;
		count++;
	    }
	}
	cerr << "Material "<<i<<"   min="<<min<<"  max="<<max<<"  count="<<count<<"\n";
    }

    Array1<int> newConds;
    newConds.add(0);
    newConds.add(1);
    newConds.add(2);
    newConds.add(4);
    newConds.add(5);
    newConds.add(3);
    newConds.add(3);
    newConds.add(3);
    newConds.add(3);
    newConds.add(3);
    newConds.add(3);
    newConds.add(3);
    newConds.add(3);
    // ...

    for (i=0; i<mesh->elems.size(); i++)
	mesh->elems[i]->cond=newConds[mesh->elems[i]->cond];

    // sort the elements based on conductivity indices
    Array1<Element *> elems=mesh->elems;
    Array1<int> sortConds;
    sortConds.add(4);
    sortConds.add(5);
    sortConds.add(1);
    sortConds.add(2);
    sortConds.add(3);
    sortConds.add(0);
    int curr=0;
    for (i=0; i<sortConds.size(); i++) {
	int cond=sortConds[i];
	for (j=0; j<elems.size(); j++)
	    if (elems[j]->cond == cond) {
		mesh->elems[curr++]=elems[j];
	    }
    }
    cerr << "\n\nAfter changing/sorting...\n";
    for (i=0; i<6; i++) {
	int first=1;
	int min=0;
	int max=0;
	int count=0;
	for (j=0; j<mesh->elems.size(); j++) {
	    if (mesh->elems[j]->cond == i) {
		if (first) { min=j; first=0; }
		max=j;
		count++;
	    }
	}
	cerr << "Material "<<i<<"   min="<<min<<"  max="<<max<<"  count="<<count<<"\n";
    }

    // done fixing stuff

    mesh->cond_tensors.resize(6);
    mesh->cond_tensors[0].resize(6);
    mesh->cond_tensors[0].initialize(0);
    mesh->cond_tensors[0][0]=mesh->cond_tensors[0][3]=mesh->cond_tensors[0][5]=AIR_CONDUCTIVITY;

    mesh->cond_tensors[1].resize(6);
    mesh->cond_tensors[1].initialize(0);
    mesh->cond_tensors[1][0]=mesh->cond_tensors[1][3]=mesh->cond_tensors[1][5]=SKIN_CONDUCTIVITY;

    mesh->cond_tensors[2].resize(6);
    mesh->cond_tensors[2].initialize(0);
    mesh->cond_tensors[2][0]=mesh->cond_tensors[2][3]=mesh->cond_tensors[2][5]=BONE_CONDUCTIVITY;

    mesh->cond_tensors[3].resize(6);
    mesh->cond_tensors[3].initialize(0);
    mesh->cond_tensors[3][0]=mesh->cond_tensors[3][3]=mesh->cond_tensors[3][5]=CSF_CONDUCTIVITY;

    mesh->cond_tensors[4].resize(6);
    mesh->cond_tensors[4].initialize(0);
    mesh->cond_tensors[4][0]=mesh->cond_tensors[4][3]=mesh->cond_tensors[4][5]=GREY_CONDUCTIVITY;
    
    mesh->cond_tensors[5].resize(6);
    mesh->cond_tensors[5].initialize(0);
    mesh->cond_tensors[5][0]=mesh->cond_tensors[5][3]=mesh->cond_tensors[5][5]=WHITE_CONDUCTIVITY;
    
    clString base2(argv[2]);
    BinaryPiostream stream2(base2, Piostream::Write);
    Pio(stream2, mesh);
    return 0;
}    
