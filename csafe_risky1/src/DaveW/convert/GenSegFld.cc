/*
 * main.c - Does everything
 *
 * David Weinstein, April 1996
 *
 */

#include <SCICore/Containers/String.h>
#include <SCICore/Persistent/Pstreams.h>
#include <DaveW/Datatypes/General/SegFld.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>

using namespace SCICore::Containers;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Datatypes;

int nx, ny, nz;

#if 0
char getVal(int i, int j, int k) {
    if (i==j && j==k && k==1) return 5; else return 0;
}

char getVal(int i, int j, int k) {
    if (i==2 || j==2 || k==2) {
	if (i==2 && k==1) {
	    if (j>=1 && j<=3) return 5+'0';
	    else return 0+'0';
	} else if (j==2 && k==2) {
	    if (i>=1 && i<=3) return 4+'0';
	    else return 0+'0';
	}
    }
    return 0+'0';
}

char getVal(int i, int j, int k) {
    if (i==3 && j==4 && (k==3 || k==5)) return '5';
    else if (i==3 && j==4 && k==4) return '4';
    else if (i==3 && j==3 && (k==3 || k==4 || k==5)) return '5';
    else if (i==3 && j==5 && (k==3 || k==4 || k==5)) return '2';
    else if ((i>=2) && (i<=6) && (j>=2) && (j<=6) && (k>=2) && (k<=6)) return '3';
    else return '0';
}

char getVal(int i, int j, int k) {
    if (i==3 && j==3 && k==3) return '5';
    else if (i==3 && j==3 && k==2) return '4';
    else return '0';
}

char getVal(int i, int j, int k) {
    double val;
    double ii=(2.*i/nx-1);
    double jj=(2.*j/ny-1);
    double kk=(2.*k/nz-1);

    val=5.2-(ii*ii+jj*jj+kk*kk)*5.4;
    if (val<0) val=0;
    char c=val+'0';
    return c;
}


#endif

char getVal(int i, int j, int k) {
    double val;
    double ii=(2.*i/nx-1);
    double jj=(2.*j/ny-1);
    double kk=(2.*k/nz-1);
    
    if (ii*ii+jj*jj+kk*kk > .8) return '0';
    if (ii*ii+(jj-.2)*(jj-.2)+kk*kk < .25) 
	if (jj<0) return '3'; else return '2';
    if (ii*ii+(jj+.2)*(jj+.2)+kk*kk < .25)
	if (jj<0) return '3'; else return '2';
    return '1';
}
    
void main(int argc, char *argv[]) {
    nx=ny=nz=33;

    ScalarFieldRGchar *sf = new ScalarFieldRGchar;
    sf->set_bounds(Point(-1,-1,-1), Point(1,1,1));
//    sf->set_bounds(Point(-.5,-.5,-.5), Point(1.5,1.5,1.5));
    sf->resize(nx,ny,nz);
    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++)
		sf->grid(i,j,k) = getVal(i,j,k);
    clString fname("test.c_sfrg");
    TextPiostream stream(fname, Piostream::Write);
    ScalarFieldHandle sfh=sf;
    Pio(stream, sfh);
}
