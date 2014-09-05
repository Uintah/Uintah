/*
 *  ModelTest: Can we build an RPI mesh?
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

/*

CC -g -n32 -mips4 -r10000  -I/home/sci/u2/dmw/MEGA/develop/model/model/1.1/include -I/home/sci/u2/dmw/MEGA/develop/model/smdModel/1.1/smdModel -I/home/sci/u2/dmw/MEGA/develop/util/util/0.3/include -I/home/sci/u2/dmw/MEGA/develop/model/mesh/4.1/include -c ModelTest.cc

CC -Wl,-woff -Wl,85 -o ModelTest -n32 -mips4 -r10000 ModelTest.o -L../lib -L/home/sci/u2/dmw/MEGA/develop/model/smdModel/1.1/lib/sgi_6 -L/home/sci/u2/dmw/MEGA/develop/util/util/0.3/lib/sgi_6 -L/home/sci/u2/dmw/MEGA/develop/parallel/comm/2.1/lib/sgi_6 -L/home/sci/u2/dmw/MEGA/develop/model/model/1.1/lib/sgi_6 -L/home/sci/u2/dmw/MEGA/develop/model/mesh/4.1/lib/sgi_6 -lsmdModel -lmodel-O -lmesh-O -lmodel-O -lutil-O -lcomm-O -lm

*/

#include "MSops.h"
#include "TopoModel.h"
#include "SmdModel.h"
#include "TFace.h"
#include "TRegion.h"

main(int argc, char **argv) {
    double px[8], py[8], pz[8];
    px[0]=0; py[0]=0; pz[0]=0;
    px[1]=1; py[1]=0; pz[1]=0;
    px[2]=0; py[2]=0; pz[2]=1;
    px[3]=0; py[3]=1; pz[3]=0;
    px[4]=.3; py[4]=.3; pz[4]=.3;
    px[5]=.5; py[5]=.3; pz[5]=.3;
    px[6]=.3; py[6]=.3; pz[6]=.5;
    px[7]=.3; py[7]=.5; pz[7]=.3;

    MD_init();
    TopoModel *model = new TopoModel("sample1");

    SSList<GEdge *> edges;
    SSList<int> edirs;
    TFace *f1 = new TFace(model, 0, edges, edirs);
    model->add(f1);

    SSList<GFace *> faces;
    SSList<int> fdirs;
    faces.append(f1);
    fdirs.append(1); 

    TRegion *r1 = new TRegion(model, 0, faces, fdirs); // define region 1;
    model->add(r1);

    // Complete the model by creating the outer shell
    model->createOuterShell();

    // now make a mesh based on the model
    pMesh mesh=MM_new(0, model);

    // make all eight of the vertices with M_createVP -- 4 for the inner
    //   tet, 4 for the outter tet
    pVertex mv1=MM_createVP(mesh, px[0], py[0], pz[0], 0, 0, f1);
    pVertex mv2=MM_createVP(mesh, px[1], py[1], pz[1], 0, 0, f1);
    pVertex mv3=MM_createVP(mesh, px[2], py[2], pz[2], 0, 0, f1);
    pVertex mv4=MM_createVP(mesh, px[3], py[3], pz[3], 0, 0, f1);

    // make all of the edges with MM_createE
    pEdge me1=MM_createE(mesh, mv1, mv2, f1);
    pEdge me2=MM_createE(mesh, mv1, mv3, f1);
    pEdge me3=MM_createE(mesh, mv1, mv4, f1);
    pEdge me4=MM_createE(mesh, mv2, mv3, f1);
    pEdge me5=MM_createE(mesh, mv2, mv4, f1);
    pEdge me6=MM_createE(mesh, mv3, mv4, f1);

    // make all of the faces with MM_createF
    pEdge pedges[3]; pedges[0]=me1; pedges[1]=me4; pedges[2]=me2;
    int pdirs[3]; pdirs[0]=1; pdirs[1]=1; pdirs[2]=0;
    pFace mf1=MM_createF(mesh, 3, pedges, pdirs, f1);
    pedges[0]=me5; pedges[1]=me6; pedges[2]=me4;
    pdirs[0]=1; pdirs[1]=0; pdirs[2]=0;
    pFace mf2=MM_createF(mesh, 3, pedges, pdirs, f1);
    pedges[0]=me3; pedges[1]=me6; pedges[2]=me2;
    pdirs[0]=1; pdirs[1]=0; pdirs[2]=0;
    pFace mf3=MM_createF(mesh, 3, pedges, pdirs, f1);
    pedges[0]=me3; pedges[1]=me5; pedges[2]=me1;
    pdirs[0]=1; pdirs[1]=0; pdirs[2]=0;
    pFace mf4=MM_createF(mesh, 3, pedges, pdirs, f1);

    // write out the mesh and the model to files
    model->writeSMD("test.smd");	// this is the model
    M_writeSMS(mesh, "test.sms", "sr");	// this is the mesh

    // also, create a .smd file
    FILE *f=fopen(".smd", "wt");
    fprintf(f, "config inputMeshName test.sms\n");
    fprintf(f, "config meshName testOut.sms\n");
    fprintf(f, "config writeMeshStats\n");
    fprintf(f, "config generateVolumeMesh 1\n");
    fprintf(f, "config constrainSurface\n");
    fprintf(f, "model size 1 0.015625 0.015625\n");
    fclose(f);

    system("/home/sci/u2/dmw/MEGA/develop/meshing/meshModel/3.9/bin/sgi_6/meshSmd\n");
    
    // ok, now load in the new model
    SGModel *model2 = new TopoModel("test.smd");
    pMesh mesh2 = MM_new(0, model2);
    M_load(mesh2, "testOut.sms");

#if 0
    sci::Mesh *smesh = new sci::Mesh;

    void *temp=0;
    double coord[3];
    int count=0;
    while(pVertex vert=M_nextVertex(mesh2, &temp)) {
	EN_setID((pEntity)vert, count++);
	V_coord(vert, coord);
	smesh->nodes.add(new sci::Node(Point(coord[0], coord[1], coord[2])));
    }

    temp=0;
    while(pRegion reg=M_nextRegion(mesh2, &temp)) {
	pPList list=R_vertices(reg);
	int idx[4];
	for (int i=0; i<4; i++) {
	    pVertex pv=(pVertex)PList_next(list, &temp);
	    idx[i]=EN_id((pEntity)pv);
	}
	smesh->elems.add(new sci::Element(smesh, idx[0], idx[1], idx[2], 
					  idx[3]));
    }
    
    TextPiostream stream("test.mesh", Piostream::Write);
    sci::MeshHandle mh=smesh;
    Pio(stream, mh);
#endif
    MD_exit();

    // use EN_setID and EN_ID AFTER writing it out -- save to lookup table
}
