

/*
 *  VDTtoMesh.cc: Convert Petr Krysl's (CalTech) VDT format to a SCIRun mesh
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

int readLine(FILE **f, char *buf) {
    char c;
    int cnt=0;
    while (!feof(*f) && ((c=fgetc(*f))!='\n')) buf[cnt++]=c;
    buf[cnt]=c;
    if (feof(*f)) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
	cerr << "usage: " << argv[0] << " basename\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;

    FILE *f=fopen(clString(clString(argv[1])+".t3d")(), "rt");
    if (!f) {
	cerr << "Error - failed to open "<<argv[1]<<".t3d\n";
	exit(0);
    }
    
    char buf[10000];
    int i, dum1, dum2, npts, ntets;
    double x, y, z;
    int i1, i2, i3, i4, c;

    // ! VDT (C) 1998 Petr Krysl
    readLine(&f, buf);
    
    // nnodes nedges ntriangles ntetras
    readLine(&f, buf);
    sscanf(buf, "%d %d %d %d", &npts, &dum1, &dum2, &ntets);
    cerr << "File has "<<npts<<" points and "<<ntets<<" tets.\n";

    Array1<Node*> allNodes(npts*3);
    for (i=0; i<npts*3; i++) allNodes[i]=new Node(Point(0,0,0));

    // etype edegree
    readLine(&f, buf);
    
    // ! points
    readLine(&f, buf);

    int cnt=0;
    // idx x y z
    for (i=0; i<npts; i++) {
	readLine(&f, buf);
	sscanf(buf, "%d %lf %lf %lf", &dum1, &x, &y, &z);
	while (dum1-1 != mesh->nodes.size()) {
	    allNodes[cnt]->p=Point(-347,-348,-349);
	    mesh->nodes.add(NodeHandle(allNodes[cnt++]));
	}
	allNodes[cnt]->p=Point(x,y,z);
	mesh->nodes.add(NodeHandle(allNodes[cnt++]));
    }

    // ! tets
    readLine(&f, buf);
    
    // idx i1 i2 i3 i4 IGNORE cond

    for (i=0; i<ntets; i++) {
	readLine(&f, buf);
	sscanf(buf, "%d %d %d %d %d %d %d", &dum1, &i1, &i2, &i3, &i4, &dum2,
	       &c);
	Element *e = new Element(mesh.get_rep(), i1-1, i2-1, i3-1, i4-1);
	e->cond=c;
	mesh->elems.add(e);
    }

    TextPiostream stream(clString(argv[1])+".mesh", Piostream::Write);
    Pio(stream, mesh);
    return 0;
}
