

#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/TriSurface.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int main(int argc, char *argv[]) {
    double z=4;
    if (argc != 2) {
	cerr << "Usage: "<<argv[0]<<" numCells\n";
	exit(0);
    }
    int num=atoi(argv[1]);
    double x=-70;
    double y=-70;

    double d=140./num;
    int num1=num+1;
    DenseMatrix *mat=new DenseMatrix(3,num1*num1);
    TriSurface *ts = new TriSurface;

    int cnt=0;
    for (; y<=70; y+=d) 
	for (x=-70.; x<=70; x+=d, cnt++) {
	    (*mat)[0][cnt]=x;
	    (*mat)[1][cnt]=y;
	    (*mat)[2][cnt]=z;
	    ts->points.add(Point(x,y,z));
	}

    int i,j;
    for (j=0; j<num; j++) 
	for (i=0; i<num; i++) {
	    ts->elements.add(new TSElement(j*num1+i, j*num1+i+1, 
					   (j+1)*num1+i));
	    ts->elements.add(new TSElement(j*num1+i+1, (j+1)*num1+i+1,
					   (j+1)*num1+i));
	}

    char name[1000];
    sprintf(name, "/local/sci/raid0/dmw/cube/plate.%d.mat", num);
    BinaryPiostream stream(name, Piostream::Write);
    MatrixHandle mh(mat);
    Pio(stream, mh);
    sprintf(name, "/local/sci/raid0/dmw/cube/plate.%d.ts", num);
    BinaryPiostream stream2(name, Piostream::Write);
    SurfaceHandle sh(ts);
    Pio(stream2, sh);
    return 1;
}
