

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

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCICore::Containers;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;

int main(int argc, char **argv)
{
    if (argc != 2) {
	cerr << "usage: " << argv[0] << " basename\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;

    ifstream ptsfile(clString(clString(argv[1])+".pts")());
    int n;
    ptsfile >> n;
    cerr << "nnodes in pts file=" << n << endl;
    while(ptsfile){
	double x,y,z;
	ptsfile >> x >> y >> z;
	if(ptsfile){
	    mesh->nodes.add(NodeHandle(new Node(Point(x,y,z))));
	}
    }
    cerr << "nnodes in mesh=" << mesh->nodes.size() << endl;

    ifstream tetrafile(clString(clString(argv[1])+".tetras")());
    int t;
    tetrafile >> t;
    cerr << "nelems in tetras files=" << t << endl;
    while(tetrafile){
	int i1, i2, i3, i4, cond;
//	tetrafile >> i1 >> i2 >> i3 >> i4;
	tetrafile >> i1 >> i2 >> i3 >> i4 >> cond;
	if(tetrafile && cond!=2){
	    Element *e = new Element(mesh.get_rep(), i1-1, i2-1, i3-1, i4-1);
	    //	    e->cond=cond;
	    e->cond=0;
	    mesh->elems.add(e);
	}
    }
    cerr << "nelems in mesh=" << mesh->elems.size() << endl;

#if 0
    ifstream condfile(clString(clString(argv[1])+".conds")());
    condfile >> t;
    cerr << "number of conductivities="<<t<< endl;
    mesh->cond_tensors.resize(t);
    for (int i=0; i<t; i++) {
	for (int j=0; j<6; j++) {
	    int dummy;
	    condfile >> dummy;
	    mesh->cond_tensors[i].add(dummy);
	}
    }

    Array1<int> numT;
    int xyz, oldSz, j;
    cerr << "Analyzing mesh...\n";
    for (i=0; i<mesh->elems.size(); i++) {
	xyz=mesh->elems[i]->cond;
	if (xyz >= numT.size()) {
	    oldSz=numT.size();
	    numT.resize(xyz+1);
	    for (j=oldSz; j<numT.size(); j++) numT[j]=0;
	}
	numT[xyz]=numT[xyz]+1;
    }
    for (i=0; i<numT.size(); i++) {
	if (numT[i]) {
	    cerr << "Mesh has "<<numT[i]<<" elements of type "<<i<<"\n";
	}
    }
#endif

    TextPiostream stream(clString(argv[1])+".mesh", Piostream::Write);
    Pio(stream, mesh);
    return 0;
}
