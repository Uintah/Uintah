
/*
 *  MakeRadialCylinder: Creates a mesh for a cylinder using radial symmetry
 *
 *
 *  Written by:
 *   Kris Zyp
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

#define NAOH_CONDUCTIVITY       0.025
#define ALUMINUM_CONDUCTIVITY   18000.000
#define INNER_CIRCLE_DIAM       19.0
// #define INNER_CIRCLE_DIAM    40.0
#define ALUM_DIAM               70.0
#define NAOH_DIAM               200.0
#define ALUMINUM_HEIGHT         40.0


using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCICore::Containers;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::Math;


// push all of the nodes into concentric circles
int wrapAroundAngle(int angle, int limit){  // allows angles to increment and wrap around when necessary
    if ((angle) == limit)       return 0;
    else return angle;
}
     
void genPtsAndTets(int nr, int nz, Mesh* mesh){
    Array3<int> nodes(nr, nr * 6, nz);
    Array1<Element *> e(3);
    Array1<int> c(6);
    Array1<float> zPos(22); 
    Array1<float> rPos(14); 
    int currIdx,parity;
    int r1,r2,angleInner,angleOuter,z;
    zPos [0] = 0;
    zPos[1] =9.5;
    zPos[2] =19.05;
    zPos[3] =29.36;
    zPos[4] =39.6875;
    zPos[5] =50;
    zPos[6] =60.325;
    zPos[7] =70;
    zPos[8] =79.375;
    zPos[9] =89.7;
    zPos[10] =100.0125;
    zPos[11] =109.53;
    zPos[12] =119.0625;
    zPos[13] =129.28;
    zPos[14] =139.7;
    zPos[15] =150.02;
    zPos[16] =160.3375;
    zPos[17] =169.85;
    zPos[18] =179.3875;
    zPos[19] =189.7;
    zPos[20] =200.025;
    zPos[21] =204.7875;

    rPos[0] = 0;
    rPos[1] = 7.62;
    rPos[2] = 7.62*2;
    rPos[3] = 7.62*3;
    rPos[4] = 7.62*4;
    rPos[5] = 7.62*5;
    rPos[6] = 7.62*6;
    rPos[7] = 7.62* 7;
    rPos[8] = 7.62* 8;
    rPos[9] = 7.62*9;
    rPos[10] = 76.2;
    rPos[11] = 82.55;
    rPos[12] = 89.65;
    rPos[13] = 96.75;

    r1 = 0;
    parity =0;
    currIdx =0;
    for (z=0; z<nz; z++) {
        mesh-> nodes.add(new Node(Point(0,0,zPos[z])));  // make center point
        nodes(0,0,z) = currIdx++;
        for (r2=1; r2<nr; r2++)
            for (angleOuter=0; angleOuter<r2*6; angleOuter++) {
                mesh->nodes.add(new Node(Point(rPos[r2]*cos(angleOuter * PI/3 /
r2),rPos[r2]*sin(angleOuter * PI/3 / r2),zPos[z]))); // create radial points
                nodes(r2,angleOuter,z) = currIdx++;
            }
    }
    for (z=0; z<nz-1; z++)
        for (r1=0; r1<nr-1; r1++) {
            r2 = r1 + 1;
            angleOuter = 0;
            angleInner = 0;
            while (angleOuter < 6 * r2) {
                parity++;
                if (angleOuter * r1 <= angleInner * r2) {  // move along both radiuses at the same rate, this one is a normal pie slice
                    c[0] = nodes(r1,wrapAroundAngle(angleInner,6*r1),z);  // the corners of the pie
                    c[1] = nodes(r1,wrapAroundAngle(angleInner,6*r1),z+1);
                    c[2] = nodes(r2,angleOuter,z);
                    c[3] = nodes(r2,angleOuter,z+1);
                    c[4] = nodes(r2,wrapAroundAngle(angleOuter+1,6*r2),z);
                    c[5] = nodes(r2,wrapAroundAngle(angleOuter+1,6*r2),z+1);
                    angleOuter++;
                }
                else {  // this one is the backwards pie slice (pointing outwards)
                    c[0] = nodes(r2,angleOuter,z);
                    c[1] = nodes(r2,angleOuter,z+1);
                    c[2] = nodes(r1,wrapAroundAngle(angleInner+1,6*r1),z);
                    c[3] = nodes(r1,wrapAroundAngle(angleInner+1,6*r1),z+1);
                    c[4] = nodes(r1,angleInner,z);
                    c[5] = nodes(r1,angleInner,z+1);
                    angleInner++;
                }
                if (parity%2) {  // use parity to keep switching direction
                    e[0] = new Element(mesh,c[0],c[2],c[3],c[4]);  // slice the pie up into tets
                    e[1] = new Element(mesh,c[1],c[3],c[4],c[5]);
                    e[2] = new Element(mesh,c[0],c[1],c[3],c[4]);
                }
                else {
                    e[0] = new Element(mesh,c[1],c[2],c[3],c[5]);
                    e[1] = new Element(mesh,c[0],c[2],c[4],c[5]);
                    e[2] = new Element(mesh,c[0],c[1],c[2],c[5]);
                }
                if (r1 < 10)
                    e[0]->cond=e[1]->cond=e[2]->cond=0;  // conductivity of inner fluid
                else    
                    if (r1 < 11)
                        e[0]->cond=e[1]->cond=e[2]->cond=1;  // conductivity of middle fluid
                    else    
                        e[0]->cond=e[1]->cond=e[2]->cond=2;  // conductivity of outer fluid
                mesh->elems.add(e[0]);  // and add them
                mesh->elems.add(e[1]);
                mesh->elems.add(e[2]);

            }
        }
}

int main(int argc, char *argv[]) {
    //    if (argc != 3) {
    //  cerr << "Usage: MakeRadialCylinder radius height\n";
    //  exit(0);
    //}
    int nr, nz;
    //nr=atoi(argv[1]);  // get radius
    //nz=atoi(argv[2]);  // get height
    Mesh *mesh = new Mesh;
    int i;
    genPtsAndTets(14,22,mesh);  // generate everything

    mesh->cond_tensors.resize(3);
    mesh->cond_tensors[0].resize(6);
    mesh->cond_tensors[0].initialize(0);
    mesh->cond_tensors[0][0]=mesh->cond_tensors[0][3]=
        mesh->cond_tensors[0][5]=0.3012;
    mesh->cond_tensors[1].resize(6);
    mesh->cond_tensors[1].initialize(0);
    mesh->cond_tensors[1][0]=mesh->cond_tensors[1][3]=
        mesh->cond_tensors[1][5]=0.1506;
    mesh->cond_tensors[2].resize(6);
    mesh->cond_tensors[2].initialize(0);
    mesh->cond_tensors[2][0]=mesh->cond_tensors[2][3]=
        mesh->cond_tensors[2][5]=0.4518;



    // clean up mesh (remove nodes with no elements)
    mesh->pack_all();
    mesh->compute_neighbors();
    for (i=0; i<mesh->nodes.size(); i++) 
        if (!mesh->nodes[i]->elems.size())
            mesh->nodes[i]=0;
    mesh->pack_all();
    mesh->compute_neighbors();
    clString fname(clString("/tmp/cyl.mesh"));
    Piostream* stream = scinew TextPiostream(fname, Piostream::Write);
    MeshHandle mH(mesh);  // save it all
    Pio(*stream, mH);
    delete(stream);
    return 1;
}
