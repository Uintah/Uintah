

/*
 *  JAStoMesh.cc: Convert .pts and .tetras to a .mesh file
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
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

int main(int argc, char **argv)
{
    if (argc != 2) {
	cerr << "usage: " << argv[0] << " basename\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;

    clString basename(argv[1]);
    ifstream ptsfile((basename+".pts")());
    int n;
    ptsfile >> n;
    cerr << "nnodes in pts file=" << n << endl;
    while(ptsfile){
	double x,y,z;
	ptsfile >> x >> y >> z;
	if(ptsfile){
	    mesh->nodes.add(Node(Point(x,y,z)));
	}
    }
    cerr << "nnodes in mesh=" << mesh->nodes.size() << endl;

    ifstream tetrafile((basename+".tetras")());
    int t;
    tetrafile >> t;
    cerr << "nelems in tetras files=" << t << endl;
    while(tetrafile){
	int i1, i2, i3, i4, cond;
//	tetrafile >> i1 >> i2 >> i3 >> i4;
	tetrafile >> i1 >> i2 >> i3 >> i4 >> cond;
	if(tetrafile && cond!=2){
	    Element *e = new Element(mesh.get_rep(), i1-1, i2-1, i3-1, i4-1);
	    e->cond=cond-1;
	    // e->cond=0;
	    mesh->elems.add(e);
	}
    }
    cerr << "nelems in mesh=" << mesh->elems.size() << endl;

    ifstream condfile((basename+".cond")());
    condfile >> t;
    cerr << "number of conductivities="<<t<< endl;
    mesh->cond_tensors.resize(t);
    int i;
    for (i=0; i<t; i++) {
	for (int j=0; j<6; j++) {
	    int dummy;
	    condfile >> dummy;
	    mesh->cond_tensors[i].add(dummy*10000);
	}
    }

    Array1<int> numT(mesh->cond_tensors.size());
    numT.initialize(0);

    cerr << "Analyzing mesh...\n";
    for (i=0; i<mesh->elems.size(); i++) {
      numT[mesh->elems[i]->cond]++;
    }
    for (i=0; i<numT.size(); i++) {
	if (numT[i]) {
	    cerr << "Mesh has "<<numT[i]<<" elements of type "<<i<<"\n";
	}
    }

    BinaryPiostream stream(basename+".mesh", Piostream::Write);
    Pio(stream, mesh);
    return 0;
}
