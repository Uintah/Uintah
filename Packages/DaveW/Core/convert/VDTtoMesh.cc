

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
#include <SCICore/Datatypes/SurfTree.h>
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
    if (argc < 2 || (argc==3)) {
	cerr << "usage: " << argv[0] << " basename [-regionconds defaultcond region0cond region1cond ...]\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;

    char buf[10000];
    int i, dum1, dum2, npts, ntets;
    double x, y, z;
    int i1, i2, i3, i4, c;
    int remap=0;
    int defaultCond;

    Array1<int> newConds;
    if (argc>2) {
	remap=1;
	newConds.resize(argc-4);
	for (i=4; i<argc; i++) newConds[i-4]=atoi(argv[i]);
	defaultCond=atoi(argv[3]);
	cerr << "DefaultCond = "<<defaultCond<<"\n";
	for (i=0; i<newConds.size(); i++)
	    cerr << "  Region "<<i<<" gets conductivity "<<newConds[i]<<"\n";
    }

    FILE *f=fopen(clString(clString(argv[1])+".t3d")(), "rt");
    if (!f) {
	cerr << "Error - failed to open "<<argv[1]<<".t3d\n";
	exit(0);
    }
    SurfaceHandle surfH;
    Piostream* surfstr=auto_istream(clString(clString(argv[1])+".st"));
    if (!surfstr)
	surfstr = auto_istream(clString(clString(argv[1])+".stree"));
    if (surfstr) Pio(*surfstr, surfH);
    SurfTree *st;
    if (surfH.get_rep() && (st=dynamic_cast<SurfTree*>(surfH.get_rep()))) {
	cerr << "Getting material indices for each component from surftree\n";
	newConds.resize(st->surfI.size());
	for (i=0; i<newConds.size(); i++) newConds[i]=st->surfI[i].matl;
    }

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
	while (dum1-1 != mesh->nodes.size())
	    mesh->nodes.add(NodeHandle(0));
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
	if (remap) {
	    if (c >= newConds.size()) e->cond=defaultCond;
	    else e->cond=newConds[c];
	}
	mesh->elems.add(e);
    }
    cerr << "Had "<<mesh->nodes.size()<<" nodes.\n";
    mesh->pack_all();
    cerr << "Now "<<mesh->nodes.size()<<" nodes.\n";

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
    
    BinaryPiostream stream(clString(argv[1])+".mesh", Piostream::Write);
    Pio(stream, mesh);
    return 0;
}
