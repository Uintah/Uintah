/*
 *  ModelTest5: Can we build an RPI mesh?
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <Containers/Array1.h>
#include <Containers/String.h>
#include <Persistent/Pstreams.h>
#include <CoreDatatypes/Mesh.h>
#include <Geometry/Point.h>
#include "MSops.h"
#include "TopoModel.h"
#include "SmdModel.h"
#include "TVertex.h"
#include "TEdge.h"
#include "TFace.h"
#include "TRegion.h"
#include <stdio.h>

#define Element		SCICore::CoreDatatypes::Element
#define Mesh		SCICore::CoreDatatypes::Mesh
#define MeshHandle	SCICore::CoreDatatypes::MeshHandle
#define Node		SCICore::CoreDatatypes::Node
#define NodeHandle	SCICore::CoreDatatypes::NodeHandle
#define Point		SCICore::Geometry::Point

using namespace SCICore::PersistentSpace;
using namespace SCICore::CoreDatatypes;

main(int argc, char **argv) {
    Array1<Point> p(8);
    p[0]=Point(0, 0, 0);
    p[1]=Point(1, 0, 0);
    p[2]=Point(0, 0, 1);
    p[3]=Point(0, 1, 0);

    MD_init();
    TopoModel *model = new TopoModel("sample1");

    TVertex *v1 = new TVertex(model, 0);
    model->add(v1);

    TEdge *e1 = new TEdge(model, 0, v1, v1);
    model->add(e1);

    SSList<GEdge *> edges;
    SSList<int> edirs;
    // Add the edge to the list defining the loops of the face - 
    // in this case it will be used in the positive sense
    edges.append(e1);
    edirs.append(1); // Used in the positive sense
    TFace *f1 = new TFace(model, 0, edges, edirs);
    model->add(f1);

    edges.clear();
    edirs.clear();
    edges.append(e1);
    edirs.append(0);
    TFace *f2 = new TFace(model, 0, edges, edirs);
    model->add(f2);

    SSList<GFace *> faces;
    SSList<int> fdirs;
    // face will be used by the shell of the region in the positve sense
    faces.append(f1);
    fdirs.append(1); // Used in the positive sense
    faces.append(f2);
    fdirs.append(1); // Used in the negative sense
    TRegion *r1 = new TRegion(model, 0, faces, fdirs); // define region 1;
    model->add(r1);

    // Complete the model by creating the outer shell
    model->createOuterShell();

    // now make a mesh based on the model
    pMesh mesh=MM_new(0, model);

    // make all four of the vertices with M_createVP
    pVertex mv1=MM_createVP(mesh, p[0].x(), p[0].y(), p[0].z(), 0, 0, v1);
    pVertex mv2=MM_createVP(mesh, p[1].x(), p[1].y(), p[1].z(), 0, 0, f2);
    pVertex mv3=MM_createVP(mesh, p[2].x(), p[2].y(), p[2].z(), 0, 0, f2);
    pVertex mv4=MM_createVP(mesh, p[3].x(), p[3].y(), p[3].z(), 0, 0, f1);

    // make all of the edges with MM_createE
    pEdge me1=MM_createE(mesh, mv1, mv2, e1);
    pEdge me2=MM_createE(mesh, mv3, mv1, e1);
    pEdge me3=MM_createE(mesh, mv1, mv4, f1);
    pEdge me4=MM_createE(mesh, mv2, mv3, e1);
    pEdge me5=MM_createE(mesh, mv2, mv4, f1);
    pEdge me6=MM_createE(mesh, mv3, mv4, f1);

    // make all of the faces with MM_createF
    pEdge pedges[3]; pedges[0]=me1; pedges[1]=me2; pedges[2]=me4;
    int pdirs[3]; pdirs[0]=0; pdirs[1]=0; pdirs[2]=0;
    pFace mf1=MM_createF(mesh, 3, pedges, pdirs, f2);
    pedges[0]=me6; pedges[1]=me5; pedges[2]=me4;
    pdirs[0]=1; pdirs[1]=0; pdirs[2]=1;
    pFace mf2=MM_createF(mesh, 3, pedges, pdirs, f1);
    pedges[0]=me3; pedges[1]=me6; pedges[2]=me2;
    pdirs[0]=1; pdirs[1]=0; pdirs[2]=1;
    pFace mf3=MM_createF(mesh, 3, pedges, pdirs, f1);
    pedges[0]=me5; pedges[1]=me3; pedges[2]=me1;
    pdirs[0]=1; pdirs[1]=0; pdirs[2]=1;
    pFace mf4=MM_createF(mesh, 3, pedges, pdirs, f1);

    // write out the mesh and the model to files
    model->writeSMD("model.smd");	// this is the model
    M_writeSMS(mesh, "model.sms", "sr");	// this is the mesh

    // also, create a .smd file
    FILE *f=fopen("model.attr", "wt");
    fprintf(f, "model.smd\n");
    fprintf(f, "config inputMeshName model.sms\n");
    fprintf(f, "config meshName model2.sms\n");
    fprintf(f, "config writeMeshStats 1\n");
    fprintf(f, "config generateVolumeMesh 1\n");
    fprintf(f, "config constrainSurface 1\n");
    fprintf(f, "model size 1 0.015625 0.015625\n");
    fclose(f);

    system("/home/sci/u2/dmw/MEGA/develop/meshing/meshModel/3.9/bin/sgi_6/meshSmd -file model.attr\n");
    
    // ok, now load in the new model
    SGModel *model2 = new TopoModel("model.smd");
    pMesh mesh2 = MM_new(0, model2);
    M_load(mesh2, "model2.sms");

    Mesh *smesh = new Mesh;

    void *temp=0;
    double coord[3];
    int count=0;
    while(pVertex vert=M_nextVertex(mesh2, &temp)) {
	EN_setID((pEntity)vert, count++);
	V_coord(vert, coord);
	smesh->nodes.add(new Node(Point(coord[0], coord[1], coord[2])));
    }

    temp=0;
    while(pRegion reg=M_nextRegion(mesh2, &temp)) {
	pPList list=R_vertices(reg);
	int idx[4];
	for (int i=0; i<4; i++) {
	    pVertex pv=(pVertex)PList_next(list, &temp);
	    idx[i]=EN_id((pEntity)pv);
	}
	smesh->elems.add(new Element(smesh, idx[0], idx[1], idx[2], 
					  idx[3]));
    }
    
    TextPiostream stream("model.mesh", Piostream::Write);
    MeshHandle mh=smesh;
    Pio(stream, mh);
    MD_exit();

    // use EN_setID and EN_ID AFTER writing it out -- save to lookup table
}
