
/*
 *  BldCubeMesh2: Generate mesh over a cube -- for Berkeley kids
 *	this time add dirichlet bc's on bottom
 *
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/Trig.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#define NAOH_CONDUCTIVITY	0.025
#define ALUMINUM_CONDUCTIVITY	18000.000
#define INNER_CIRCLE_DIAM	19.0
// #define INNER_CIRCLE_DIAM	40.0
#define ALUM_DIAM		70.0
#define NAOH_DIAM		200.0
#define ALUMINUM_HEIGHT		40.0

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCICore::Containers;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::Math;

Array1<double> zPos;
Array1<double> rNaOH;
Array1<double> rAlum;
int midz, midr, posBdryR, negBdryR;

// push all of the nodes into concentric circles
Point nodePos(int i, int j, int k) {
    if (i==midr && j==midr) return Point(0, 0, zPos[k]);
    double x,y;
    x=i-midr;
    y=j-midr;
    double r;
    int m=Max(Abs(x), Abs(y))+midr;
    if (k<=midz) {
	r=rAlum[m];
    } else {
	r=rNaOH[m];
    }	
    double d=Sqrt(x*x+y*y);
    return Point(x*r/d, y*r/d, zPos[k]);
}

Point nodePos2(int i, int j) {
    double x,y;
    x=i-midr;
    y=j-midr;
    double r;
    int m=Max(Abs(x), Abs(y))+midr;
    r=rNaOH[m];
    double d=Sqrt(x*x+y*y);
    return Point(x*r/d, y*r/d, 0);
}


void genPtsAndTets(int nr, int nz, Point min, Vector diag, Mesh* mesh){
    int currIdx=0;
    Array3<int> nodes(nr, nr, nz);

    int i,j,k;
    for (k=0; k<nz; k++)
	for (j=0; j<nr; j++)
	    for (i=0; i<nr; i++) {
		nodes(i,j,k)=currIdx++;
		mesh->nodes.add(new Node(nodePos(i,j,k)));
	    }

    Array1<Element *> e(5);
    Array1<int> c(8);
    for (k=0; k<nz-1; k++) {
	for (j=0; j<nr-1; j++) {
	    for (i=0; i<nr-1; i++) {
//		if (k<midz && ((i+1)<nr/7 || (nr-i-1)<nr/7 ||
//				(j+1)<nr/7 || (nr-j-1)<nr/7)) continue;
		c[0]=nodes(i,j,k);
		c[1]=nodes(i+1,j,k);
		c[2]=nodes(i+1,j+1,k);
		c[3]=nodes(i,j+1,k);
		c[4]=nodes(i,j,k+1);
		c[5]=nodes(i+1,j,k+1);
		c[6]=nodes(i+1,j+1,k+1);
		c[7]=nodes(i,j+1,k+1);
		if ((i+j+k)%2) {
		    e[0]=new Element(mesh, c[0], c[1], c[2], c[5]);
		    e[1]=new Element(mesh, c[0], c[2], c[3], c[7]);
		    e[2]=new Element(mesh, c[0], c[2], c[5], c[7]);
		    e[3]=new Element(mesh, c[0], c[4], c[5], c[7]);
		    e[4]=new Element(mesh, c[2], c[5], c[6], c[7]);
		} else {
		    e[0]=new Element(mesh, c[1], c[0], c[3], c[4]);
		    e[1]=new Element(mesh, c[1], c[3], c[2], c[6]);
		    e[2]=new Element(mesh, c[1], c[3], c[4], c[6]);
		    e[3]=new Element(mesh, c[1], c[5], c[4], c[6]);
		    e[4]=new Element(mesh, c[3], c[4], c[7], c[6]);
		}
		e[0]->cond=e[1]->cond=e[2]->cond=e[3]->cond=e[4]->cond=
		    (k<midz);
		mesh->elems.add(e[0]); 
		mesh->elems.add(e[1]); 
		mesh->elems.add(e[2]); 
		mesh->elems.add(e[3]); 
		mesh->elems.add(e[4]); 
	    }
	}
    }
}

void main(int argc, char *argv[]) {
    if (argc != 3) {
	cerr << "Usage: "<<argv[1]<<" nr nz\n";
	exit(0);
    }
    int nr, nz;
    nr=atoi(argv[1]);
    midr=nr-1;
    nr=nr*2-1;
    nz=atoi(argv[2]);
    midz=nz-1;
    nz=nz*2-1;

    zPos.resize(nz);
    zPos[0]=-.8;
    zPos[midz]=0;
    zPos[nz-1]=2;
    int i,j;
    double dz=.8/midz;
    for (i=1; i<midz; i++) zPos[i]=-.8+i*dz;
    dz=2./(nz-1-midz);
    for (i=midz+1; i<nz-1; i++) zPos[i]=(i-midz)*dz;

    rNaOH.resize(nr);
    rAlum.resize(nr);
    negBdryR=midr/2;
    posBdryR=nr-1-negBdryR;
    rAlum[midr]=rNaOH[midr]=0;
    rAlum[posBdryR]=rNaOH[posBdryR]=(INNER_CIRCLE_DIAM / 2.0);
    rAlum[nr-1]=(ALUM_DIAM / 2.0);
    rNaOH[nr-1]=(NAOH_DIAM / 2.0);
    double dr = rAlum[posBdryR] / (posBdryR-midr);
    for (i=midr+1; i<posBdryR; i++) rAlum[i]=rNaOH[i]=(i-midr)*dr;
    dr=(rAlum[nr-1]-rAlum[posBdryR]) / (nr-1-posBdryR);
    for (i=posBdryR+1; i<nr-1; i++) rAlum[i]=(i-posBdryR)*dr+rAlum[posBdryR];
    dr=(rNaOH[nr-1]-rNaOH[posBdryR]) / (nr-1-posBdryR);
    for (i=posBdryR+1; i<nr-1; i++) rNaOH[i]=(i-posBdryR)*dr+rNaOH[posBdryR];
    for (i=0; i<midr; i++) {
	rAlum[i]=-rAlum[nr-1-i];
	rNaOH[i]=-rNaOH[nr-1-i];
    }
    
    cerr << "Alum r: ";
    for (i=0; i<nr; i++) {
	cerr << rAlum[i]<<" ";
    }
    cerr << "\nNaOH r: ";
    for (i=0; i<nr; i++) {
	cerr << rNaOH[i]<<" ";
    }
    cerr << "\nz: ";
    for (i=0; i<nz; i++) {
	cerr << zPos[i]<<" ";
    }
    cerr <<"\nposBdryR="<<posBdryR<<" negBdryR="<<negBdryR<<" midr="<<midr<<" midz="<<midz<<"\n";

    Point min(-100,-100,-.8);
    Vector diag(200,200,2.8);
    Mesh *mesh = new Mesh;
    mesh->cond_tensors.resize(2);

    // NaOH
    mesh->cond_tensors[0].resize(6);
    mesh->cond_tensors[0].initialize(0);
    mesh->cond_tensors[0][0]=mesh->cond_tensors[0][3]=
	mesh->cond_tensors[0][5]=0.025;

    // Aluminum
    mesh->cond_tensors[1].resize(6);
    mesh->cond_tensors[1].initialize(0);
    mesh->cond_tensors[1][0]=mesh->cond_tensors[1][3]=
	mesh->cond_tensors[1][5]=18000;

    genPtsAndTets(nr,nz,min,diag,mesh);

    // set PotentialDifference boundary conditions
    int count=0;
    int nocount=0;
    for (j=negBdryR; j<=posBdryR; j++)
	for (i=negBdryR; i<=posBdryR; i++) {
	    int idx=nr*nr*midz+nr*j+i;
		Point p(mesh->nodes[idx]->p);
	    mesh->nodes[idx]->pdBC = new 
		PotentialDifferenceBC(idx-nr*nr, p.x()/(INNER_CIRCLE_DIAM/2.));
//	    mesh->nodes[i]->bc = new 
//		DirichletBC(0, p.x()/(INNER_CIRCLE_DIAM/2.));
	    count++;
	}


//    pin the bottom middle node (aluminum) to zero
//    mesh->nodes[nr*(nr/2)+nr/2]->bc = new DirichletBC(0,0);
//    mesh->nodes[nr*nr*(midz-1)+nr*(nr/2)+nr/2]->bc = new DirichletBC(0,0);

    // make duplicate nodes
    Array1<int> map(nr*nr*nz);
    map.initialize(-1);
    for (j=0; j<nr; j++)
	for (i=0; i<nr; i++) {
	    if (!mesh->nodes[nr*nr*midz+nr*j+i]->pdBC) {
//	    if (!mesh->nodes[nr*nr*midz+nr*j+i]->bc) {
		map[nr*nr*midz+nr*j+i]=mesh->nodes.size();
		mesh->nodes.add(new Node(nodePos2(i,j)));
		nocount++;
	    }
	}
    
    int links=0;
    // make NaOH (cond==0) elements point to duplicate nodes
    for (i=0; i<mesh->elems.size(); i++) 
	if (mesh->elems[i]->cond==0) {
	    for (j=0; j<4; j++) 
		if (map[mesh->elems[i]->n[j]] != -1) {
		    mesh->elems[i]->n[j] = map[mesh->elems[i]->n[j]];
		    links++;
		}
	}

    // clean up mesh (remove nodes with no elements)
    mesh->pack_all();
    mesh->compute_neighbors();
    for (i=0; i<mesh->nodes.size(); i++) 
	if (!mesh->nodes[i]->elems.size())
	    mesh->nodes[i]=0;
    mesh->pack_all();
    mesh->compute_neighbors();
    cerr <<"Mesh has "<<mesh->nodes.size()<<" nodes with "<<count<<" pairs, and we fixed "<<links<<" links and had "<<nocount<<" insulation nodes.\n";

    clString fname(clString("/local/sci/raid0/dmw/cube/cube.")+
		   clString(argv[1])+clString("-")+clString(argv[2])+
		   clString(".thin.mesh"));
    Piostream* stream = scinew TextPiostream(fname, Piostream::Write);
    MeshHandle mH(mesh);
    Pio(*stream, mH);
    clString fname2(clString("/local/sci/raid0/dmw/cube/cube.")+
		    clString(argv[1])+clString("-")+clString(argv[2])+
		    clString(".thick.mesh"));
    Piostream* stream2 = scinew TextPiostream(fname2, Piostream::Write);
    for (i=0; i<mesh->nodes.size(); i++) 
	mesh->nodes[i]->p.z(mesh->nodes[i]->p.z()*50);
    Pio(*stream2, mH);
    delete(stream);
    delete(stream2);
}
