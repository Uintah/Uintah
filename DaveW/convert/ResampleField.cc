/*
 *  ResampleField.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Containers/Array2.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Persistent/Pstreams.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::PersistentSpace;
using namespace std;

void boxFilter();
void padFld();
void voteFilter();
void voteFilterWeighted();
void triangleFilter();
void buildUndersampleTriangleTable(Array2<double>*, int, int, double);
void buildOversampleTriangleTable(Array2<double>*, int, double);
double padD;
float padF;
int padI;
uchar padU;
char padC;

ScalarFieldRGBase *isf, *osf, *fldX, *fldY;
void histo(ScalarFieldRGchar *sfc, int nx, int ny, int nz) {
  Array1<int> bins(6);
  bins.initialize(0);
  int i, j, k;
  char c;
  for (i=0; i<nx; i++)
    for (j=0; j<ny; j++)
      for (k=0; k<nz; k++) {
	c = sfc->grid(i,j,k);
	if (c >= '0' && c <= '5')
	  bins[c-'0']++;
      }
  cerr << "Bins: \n";
  for (i=0; i<bins.size(); i++)
    cerr << "    "<<i<<" "<<bins[i]<<"\n";
  cerr << "\n";
}

int main(int argc, char *argv[]) {
    if (argc != 7 && argc != 8) {
	cerr << "Usage: "<<argv[0]<<" infile outfile {box|triangle|vote|wvote} nx ny nz [-pad]\n";
	return 0;
    }
    clString inname(argv[1]);
    clString outname(argv[2]);
    clString ftype(argv[3]);
    int nx=atoi(argv[4]);
    int ny=atoi(argv[5]);
    int nz=atoi(argv[6]);
    if (nx<1 || ny<1 || nz<1) {
	cerr << "Error -- bad output dimensions.\n";
	return -1;
    }
    
    if (ftype != "box" && ftype != "triangle" && ftype != "vote" && ftype != "wvote") {
	cerr << "Error: bad filter type "<<ftype<<"\n";
	return -1;
    }

    ScalarFieldHandle ifh;
    Piostream* stream=auto_istream(inname);
    if (!stream) {
	cerr << "Error: couldn't open "<<inname<<".\n";
	return -1;
    }
    Pio(*stream, ifh);
    if (!ifh.get_rep()) {
	cerr << "Error reading field "<<inname<<".\n";
	return -1;
    }

    isf=ifh->getRGBase();
    if(!isf){
	cerr << "FieldFilter can't deal with unstructured grids.\n";
	return -1;
    }
    Point p1, p2;
    ifh->get_bounds(p1,p2);

    ScalarFieldRGdouble *ifd=isf->getRGDouble();
    ScalarFieldRGfloat *iff=isf->getRGFloat();
    ScalarFieldRGint *ifi=isf->getRGInt();
    ScalarFieldRGuchar *ifu=isf->getRGUchar();
    ScalarFieldRGshort *ifs=isf->getRGShort();
    ScalarFieldRGchar *ifc=isf->getRGChar();

    ScalarFieldHandle oFldHandle;

    if (ifd) {
	ScalarFieldRGdouble *fX, *fY, *of;
	fX=new ScalarFieldRGdouble();
	fX->resize(nx, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGdouble();
	fY->resize(nx, ny, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGdouble();
	of->resize(nx, ny, nz);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
	padD=ifd->grid(0,0,0);
    } else if (iff) {
	ScalarFieldRGfloat *fX, *fY, *of;
	fX=new ScalarFieldRGfloat();
	fX->resize(nx, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGfloat();
	fY->resize(nx, ny, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGfloat();
	of->resize(nx, ny, nz);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
	padF=iff->grid(0,0,0);
    } else if (ifi) {
	ScalarFieldRGint *fX, *fY, *of;
	fX=new ScalarFieldRGint();
	fX->resize(nx, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGint();
	fY->resize(nx, ny, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGint();
	of->resize(nx, ny, nz);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
	padI=ifi->grid(0,0,0);
    } else if (ifs) {
	ScalarFieldRGshort *fX, *fY, *of;
	fX=new ScalarFieldRGshort();
	fX->resize(nx, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGshort();
	fY->resize(nx, ny, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGshort();
	of->resize(nx, ny, nz);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
	padI=ifs->grid(0,0,0);
    } else if (ifc) {
        histo(ifc, ifc->nx, ifc->ny, ifc->nz);
	ScalarFieldRGchar *fX, *fY, *of;
	fX=new ScalarFieldRGchar();
	fX->resize(nx, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGchar();
	fY->resize(nx, ny, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGchar();
	of->resize(nx, ny, nz);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
	padC=ifc->grid(0,0,0);
    } else {					// must be uchar field
	ScalarFieldRGuchar *fX, *fY, *of;
	fX=new ScalarFieldRGuchar();
	fX->resize(nx, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGuchar();
	fY->resize(nx, ny, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGuchar();
	of->resize(nx, ny, nz);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
	padU=ifu->grid(0,0,0);
    }
    fldX->set_bounds(p1, p2);
    fldY->set_bounds(p1, p2);
    osf->set_bounds(p1, p2);
    if (ftype == "box") {
	boxFilter();
    } else if (ftype == "triangle") {
	triangleFilter();
    } else if (ftype == "vote") {
	voteFilter();
    } else {	// "wvote"
	cerr << "Using weighted filter.\n";
	voteFilterWeighted();
    }

    // gotta 0 pad the field!
    if (argc == 8) {
	padFld();
    }

    BinaryPiostream stream2(outname, Piostream::Write);
//    oFldHandle->set_raw(1);
    Pio(stream2, oFldHandle);
    return 1;
}

#if 0
void padFld() {
    ScalarFieldRGdouble *fd;
    ScalarFieldRGfloat *ff;
    ScalarFieldRGint *fi;
    ScalarFieldRGuchar *fc;
    fd=osf->getRGDouble();
    ff=osf->getRGFloat();
    fi=osf->getRGInt();
    fc=osf->getRGUchar();
    Point min, max;
    osf->get_bounds(min,max);
    int nx=osf->nx; 
    int ny=osf->ny;
    int nz=osf->nz;
    Vector d(max-min);
    Point newMin, newMax;
    d.x(d.x()*(nx+1)/nx);
    d.y(d.y()*(ny+1)/ny);
    d.z(d.z()*(nz+1)/nz);
    osf->set_bounds(max-d, min+d);
    osf->nx=nx+2;
    osf->ny=ny+2;
    osf->nz=nz+2;
    if (fd) {
	Array3<double> oldGrid(fd->grid);
	for (int ii=0; ii<nx; ii++)
	    for (int jj=0; jj<ny; jj++)
		for (int kk=0; kk<nz; kk++)
		    oldGrid(ii,jj,kk)=fd->grid(ii,jj,kk);
	fd->grid.newsize(nx+2, ny+2, nz+2);
	fd->grid.initialize(padD);
	printf("Padding with %lf...\n", padD);
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    fd->grid(i+1,j+1,k+1)=oldGrid(i,j,k);
    } else if (ff) {
	Array3<float> oldGrid(ff->grid);
	for (int ii=0; ii<nx; ii++)
	    for (int jj=0; jj<ny; jj++)
		for (int kk=0; kk<nz; kk++)
		    oldGrid(ii,jj,kk)=ff->grid(ii,jj,kk);
	ff->grid.newsize(nx+2, ny+2, nz+2);
	ff->grid.initialize(padF);
	printf("Padding with %f...\n", padF);
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    ff->grid(i+1,j+1,k+1)=oldGrid(i,j,k);
    } else if (fi) {
	Array3<int> oldGrid(fi->grid);
	for (int ii=0; ii<nx; ii++)
	    for (int jj=0; jj<ny; jj++)
		for (int kk=0; kk<nz; kk++)
		    oldGrid(ii,jj,kk)=fi->grid(ii,jj,kk);
	fi->grid.newsize(nx+2, ny+2, nz+2);
	fi->grid.initialize(padI);
	printf("Padding with %d...\n", padI);
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    fi->grid(i+1,j+1,k+1)=oldGrid(i,j,k);
    } else if (fc) {
	Array3<uchar> oldGrid(fc->grid);
	for (int ii=0; ii<nx; ii++)
	    for (int jj=0; jj<ny; jj++)
		for (int kk=0; kk<nz; kk++)
		    oldGrid(ii,jj,kk)=fc->grid(ii,jj,kk);
	fc->grid.newsize(nx+2, ny+2, nz+2);
	fc->grid.initialize(padC);
	printf("Padding with %d...\n", padC);
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    fc->grid(i+1,j+1,k+1)=oldGrid(i,j,k);
    } else {
	cerr << "Unknown ScalarFieldRGBase type in ResampleField!\n";
    }
}
#endif

void padFld() {
    ScalarFieldRGdouble *fd;
    ScalarFieldRGfloat *ff;
    ScalarFieldRGint *fi;
    ScalarFieldRGuchar *fu;
    ScalarFieldRGchar *fc;
    fd=osf->getRGDouble();
    ff=osf->getRGFloat();
    fi=osf->getRGInt();
    fc=osf->getRGChar();
    fu=osf->getRGUchar();
    int nx=osf->nx; 
    int ny=osf->ny;
    int nz=osf->nz;
    if (fd) {
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    if (i==0 || j==0 || k==0 || i==(nx-1) || j==(ny-1) || 
			k==(nz-1)) 
			fd->grid(i,j,k)=padD;
    } else if (ff) {
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    if (i==0 || j==0 || k==0 || i==(nx-1) || j==(ny-1) || 
			k==(nz-1)) 
			fd->grid(i,j,k)=padF;
    } else if (fi) {
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    if (i==0 || j==0 || k==0 || i==(nx-1) || j==(ny-1) || 
			k==(nz-1)) 
			fi->grid(i,j,k)=padI;
    } else if (fc) {
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    if (i==0 || j==0 || k==0 || i==(nx-1) || j==(ny-1) || 
			k==(nz-1)) 
			fc->grid(i,j,k)=padC;
    } else if (fu) {
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++)
		    if (i==0 || j==0 || k==0 || i==(nx-1) || j==(ny-1) || 
			k==(nz-1)) 
			fu->grid(i,j,k)=padU;
    } else {
	cerr << "Unknown ScalarFieldRGBase type in ResampleField!\n";
    }
}

void boxFilter() {
    ScalarFieldRGdouble *ifd, *xfd, *yfd, *ofd;
    ScalarFieldRGfloat *iff, *xff, *yff, *off;
    ScalarFieldRGint *ifi, *xfi, *yfi, *ofi;
    ScalarFieldRGshort *ifs, *xfs, *yfs, *ofs;
    ScalarFieldRGchar *ifc, *xfc, *yfc, *ofc;
    ScalarFieldRGuchar *ifu, *xfu, *yfu, *ofu;
    ifd=isf->getRGDouble();
    xfd=fldX->getRGDouble();
    yfd=fldY->getRGDouble();
    ofd=osf->getRGDouble();
    iff=isf->getRGFloat();
    xff=fldX->getRGFloat();
    yff=fldY->getRGFloat();
    off=osf->getRGFloat();
    ifi=isf->getRGInt();
    xfi=fldX->getRGInt();
    yfi=fldY->getRGInt();
    ofi=osf->getRGInt();
    ifs=isf->getRGShort();
    xfs=fldX->getRGShort();
    yfs=fldY->getRGShort();
    ofs=osf->getRGShort();
    ifc=isf->getRGChar();
    xfc=fldX->getRGChar();
    yfc=fldY->getRGChar();
    ofc=osf->getRGChar();
    ifu=isf->getRGUchar();
    xfu=fldX->getRGUchar();
    yfu=fldY->getRGUchar();
    ofu=osf->getRGUchar();
    
    double Xratio=1./((osf->nx-1.)/(isf->nx-1.));
cerr << "XRatio = " <<Xratio<<"\n";
    double curr=0;
    for (int i=0; i<fldX->nx; i++, curr+=Xratio) {
	for (int j=0, jj=0; j<fldX->ny; j++, jj++) {
	    for (int k=0, kk=0; k<fldX->nz; k++, kk++) {
		if (ifd)
		    xfd->grid(i,j,k)=ifd->grid((int)curr+0,jj,kk);
		else if (iff)
		    xff->grid(i,j,k)=iff->grid((int)curr+0,jj,kk);
		else if (ifi)
		    xfi->grid(i,j,k)=ifi->grid((int)curr+0,jj,kk);
		else if (ifs)
		    xfs->grid(i,j,k)=ifs->grid((int)curr+0,jj,kk);
		else if (ifu)
		    xfu->grid(i,j,k)=ifu->grid((int)curr+0,jj,kk);
		else 
		    xfc->grid(i,j,k)=ifc->grid((int)curr+0,jj,kk);
	    }
	}
    }
    double Yratio=1./((osf->ny-1.)/(isf->ny-1.));
    curr=0;
cerr << "YRatio = " <<Yratio<<"\n";
    for (int j=0; j<fldY->ny; j++, curr+=Yratio) {
	for (int i=0, ii=0; i<fldY->nx; i++, ii++) {
	    for (int k=0, kk=0; k<fldY->nz; k++, kk++) {
		if (ifd)
		    yfd->grid(i,j,k)=xfd->grid(ii,(int)curr+0,kk);
		else if (iff)
		    yff->grid(i,j,k)=xff->grid(ii,(int)curr+0,kk);
		else if (ifi)
		    yfi->grid(i,j,k)=xfi->grid(ii,(int)curr+0,kk);
		else if (ifs)
		    yfs->grid(i,j,k)=xfs->grid(ii,(int)curr+0,kk);
		else if (ifu)
		    yfu->grid(i,j,k)=xfu->grid(ii,(int)curr+0,kk);
		else 
		    yfc->grid(i,j,k)=xfc->grid(ii,(int)curr+0,kk);
	    }
	}
    }
    double Zratio=1./((osf->nz-1.)/(isf->nz-1.));
    curr=0;
cerr << "ZRatio = " <<Zratio<<"\n";
    for (int k=0; k<osf->nz; k++, curr+=Zratio) {
	for (int i=0, ii=0; i<osf->nx; i++, ii++) {
	    for (int j=0, jj=0; j<osf->ny; j++, jj++) {
		if (ifd)
		    ofd->grid(i,j,k)=yfd->grid(ii,jj,(int)curr+0);
		else if (iff)
		    off->grid(i,j,k)=yff->grid(ii,jj,(int)curr+0);
		else if (ifi)
		    ofi->grid(i,j,k)=yfi->grid(ii,jj,(int)curr+0);
		else if (ifs)
		    ofs->grid(i,j,k)=yfs->grid(ii,jj,(int)curr+0);
		else if (ifu)
		    ofu->grid(i,j,k)=yfu->grid(ii,jj,(int)curr+0);
		else 
		    ofc->grid(i,j,k)=yfc->grid(ii,jj,(int)curr+0);
	    }	
	}
    }
}

void triangleFilter() {
    ScalarFieldRGdouble *ifd, *xfd, *yfd, *ofd;
    ScalarFieldRGfloat *iff, *xff, *yff, *off;
    ScalarFieldRGint *ifi, *xfi, *yfi, *ofi;
    ScalarFieldRGshort *ifs, *xfs, *yfs, *ofs;
    ScalarFieldRGchar *ifc, *xfc, *yfc, *ofc;
    ScalarFieldRGuchar *ifu, *xfu, *yfu, *ofu;
    ifd=isf->getRGDouble();
    xfd=fldX->getRGDouble();
    yfd=fldY->getRGDouble();
    ofd=osf->getRGDouble();
    iff=isf->getRGFloat();
    xff=fldX->getRGFloat();
    yff=fldY->getRGFloat();
    off=osf->getRGFloat();
    ifi=isf->getRGInt();
    xfi=fldX->getRGInt();
    yfi=fldY->getRGInt();
    ofi=osf->getRGInt();
    ifs=isf->getRGShort();
    xfs=fldX->getRGShort();
    yfs=fldY->getRGShort();
    ofs=osf->getRGShort();
    ifc=isf->getRGChar();
    xfc=fldX->getRGChar();
    yfc=fldY->getRGChar();
    ofc=osf->getRGChar();
    ifu=isf->getRGUchar();
    xfu=fldX->getRGUchar();
    yfu=fldY->getRGUchar();
    ofu=osf->getRGUchar();
    
    double Xratio=(osf->nx-1.)/(isf->nx-1.);
    if (Xratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<fldX->nx; i++, ii++)
	    for (int j=0, jj=0; j<fldX->ny; j++, jj++)
		for (int k=0, kk=0; k<fldX->nz; k++, kk++)
		    if (ifd)
			xfd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			xff->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			xfi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else if (ifs)
			xfs->grid(i,j,k)=ifs->grid(ii, jj, kk);
		    else if (ifu)
			xfu->grid(i,j,k)=ifu->grid(ii, jj, kk);
		    else 
			xfc->grid(i,j,k)=ifc->grid(ii, jj, kk);
    } else if (Xratio<1) {		// undersampling     big->small
	int span=ceil(2./Xratio);
	Array2<double> table(fldX->nx, span);
	buildUndersampleTriangleTable(&table, fldX->nx, span, Xratio);
	for (int i=0; i<fldX->nx; i++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(i,l);
		int inPixelIdx=(int)((i-1)/Xratio+0+l);
		if (inPixelIdx<0) inPixelIdx=0;
		else if (inPixelIdx>isf->nx-1) inPixelIdx=isf->nx-1;
		for (int j=0, jj=0; j<fldX->ny; j++, jj++)
		    for (int k=0, kk=0; k<fldX->nz; k++, kk++) 
			if (ifd)
			    xfd->grid(i,j,k)+=ifd->grid(inPixelIdx,jj,kk)*
				tEntry;
			else if (iff)
			    xff->grid(i,j,k)+=iff->grid(inPixelIdx,jj,kk)*
				tEntry;
			else if (ifi)
			    xfi->grid(i,j,k)+=ifi->grid(inPixelIdx,jj,kk)*
				tEntry;
			else if (ifs)
			    xfs->grid(i,j,k)+=ifs->grid(inPixelIdx,jj,kk)*
				tEntry;
			else if (ifu)
			    xfu->grid(i,j,k)+=ifu->grid(inPixelIdx,jj,kk)*
				tEntry;
			else 
			    xfc->grid(i,j,k)+=ifc->grid(inPixelIdx,jj,kk)*
				tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(fldX->nx, 2);
	buildOversampleTriangleTable(&table, fldX->nx, Xratio);
	for (int i=0; i<fldX->nx; i++) {
	    int left=floor(i/Xratio)+0;
	    int right=ceil(i/Xratio)+0;
	    double lEntry=table(i,0);
	    double rEntry=table(i,1);
	    for (int j=0, jj=0; j<fldX->ny; j++, jj++)
		for (int k=0, kk=0; k<fldX->nz; k++, kk++)
		    if (ifd)
			xfd->grid(i,j,k)=ifd->grid(left,jj,kk)*lEntry+
			    ifd->grid(right,jj,kk)*rEntry;
		    else if (iff)
			xff->grid(i,j,k)=iff->grid(left,jj,kk)*lEntry+
			    iff->grid(right,jj,kk)*rEntry;
		    else if (ifi)
			xfi->grid(i,j,k)=ifi->grid(left,jj,kk)*lEntry+
			    ifi->grid(right,jj,kk)*rEntry;
		    else if (ifs)
			xfs->grid(i,j,k)=ifs->grid(left,jj,kk)*lEntry+
			    ifs->grid(right,jj,kk)*rEntry;
		    else if (ifu)
			xfu->grid(i,j,k)=ifu->grid(left,jj,kk)*lEntry+
			    ifu->grid(right,jj,kk)*rEntry;
		    else 
			xfc->grid(i,j,k)=ifc->grid(left,jj,kk)*lEntry+
			    ifc->grid(right,jj,kk)*rEntry;
	}
    }
    double Yratio=(fldY->ny-1.)/(isf->ny-1.);
    if (Yratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<fldY->nx; i++, ii++)
	    for (int j=0, jj=0; j<fldY->ny; j++, jj++)
		for (int k=0, kk=0; k<fldY->nz; k++, kk++)
		    if (ifd)
			yfd->grid(i,j,k)=xfd->grid(ii, jj, kk);
		    else if (iff)
			yff->grid(i,j,k)=xff->grid(ii, jj, kk);
		    else if (ifi)
			yfi->grid(i,j,k)=xfi->grid(ii, jj, kk);
		    else if (ifs)
			yfs->grid(i,j,k)=xfs->grid(ii, jj, kk);
		    else if (ifu)
			yfu->grid(i,j,k)=xfu->grid(ii, jj, kk);
		    else 
			yfc->grid(i,j,k)=xfc->grid(ii, jj, kk);
    } else if (Yratio<1) {		// undersampling     big->small
	int span=ceil(2./Yratio);
	Array2<double> table(fldY->ny, span);
	buildUndersampleTriangleTable(&table, fldY->ny, span, Yratio);
	for (int j=0; j<fldY->ny; j++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(j,l);
		int inPixelIdx=(int)((j-1)/Yratio+0+l);
		if (inPixelIdx<0) inPixelIdx=0;
		else if (inPixelIdx>isf->ny-1) inPixelIdx=isf->ny-1;
		for (int i=0, ii=0; i<fldY->nx; i++, ii++)
		    for (int k=0, kk=0; k<fldY->nz; k++, kk++) 
			if (ifd)
			    yfd->grid(i,j,k)+=xfd->grid(ii,inPixelIdx,kk)*
				tEntry;
			else if (iff)
			    yff->grid(i,j,k)+=xff->grid(ii,inPixelIdx,kk)*
				tEntry;
			else if (ifi)
			    yfi->grid(i,j,k)+=xfi->grid(ii,inPixelIdx,kk)*
				tEntry;
			else if (ifs)
			    yfs->grid(i,j,k)+=xfs->grid(ii,inPixelIdx,kk)*
				tEntry;
			else if (ifu)
			    yfu->grid(i,j,k)+=xfu->grid(ii,inPixelIdx,kk)*
				tEntry;
			else 
			    yfc->grid(i,j,k)+=xfc->grid(ii,inPixelIdx,kk)*
				tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(fldY->ny, 2);
	buildOversampleTriangleTable(&table, fldY->ny, Yratio);
	for (int j=0; j<fldY->ny; j++) {
	    int left=floor(j/Yratio)+0;
	    int right=ceil(j/Yratio)+0;
	    double lEntry=table(j,0);
	    double rEntry=table(j,1);
	    for (int i=0, ii=0; i<fldY->nx; i++, ii++)
		for (int k=0, kk=0; k<fldY->nz; k++, kk++)
		    if (ifd)
			yfd->grid(i,j,k)=xfd->grid(ii,left,kk)*lEntry+
			    xfd->grid(ii,right,kk)*rEntry;
		    else if (iff)
			yff->grid(i,j,k)=xff->grid(ii,left,kk)*lEntry+
			    xff->grid(ii,right,kk)*rEntry;
		    else if (ifi)
			yfi->grid(i,j,k)=xfi->grid(ii,left,kk)*lEntry+
			    xfi->grid(ii,right,kk)*rEntry;
		    else if (ifs)
			yfs->grid(i,j,k)=xfs->grid(ii,left,kk)*lEntry+
			    xfs->grid(ii,right,kk)*rEntry;
		    else if (ifu)
			yfu->grid(i,j,k)=xfu->grid(ii,left,kk)*lEntry+
			    xfu->grid(ii,right,kk)*rEntry;
		    else 
			yfc->grid(i,j,k)=xfc->grid(ii,left,kk)*lEntry+
			    xfc->grid(ii,right,kk)*rEntry;
	}
    }
    double Zratio=(osf->nz-1.)/(isf->nz-1.);
    if (Zratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<osf->nx; i++, ii++)
	    for (int j=0, jj=0; j<osf->ny; j++, jj++)
		for (int k=0, kk=0; k<osf->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=yfd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=yff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=yfi->grid(ii, jj, kk);
		    else if (ifs)
			ofs->grid(i,j,k)=yfs->grid(ii, jj, kk);
		    else if (ifu)
			ofu->grid(i,j,k)=yfu->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=yfc->grid(ii, jj, kk);
    } else if (Zratio<1) {		// undersampling     big->small
	int span=ceil(2./Zratio);
	Array2<double> table(osf->nz, span);
	buildUndersampleTriangleTable(&table, osf->nz, span, Zratio);
	for (int k=0; k<osf->nz; k++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(k,l);
		int inPixelIdx=(int)((k-1)/Zratio+0+l);
		if (inPixelIdx<0) inPixelIdx=0;
		else if (inPixelIdx>isf->nz-1) inPixelIdx=isf->nz-1;
		for (int i=0, ii=0; i<osf->nx; i++, ii++) 
		    for (int j=0, jj=0; j<osf->ny; j++, jj++)
			if (ifd)
			    ofd->grid(i,j,k)+=yfd->grid(ii,jj,inPixelIdx)*
				tEntry;
			else if (iff)
			    off->grid(i,j,k)+=yff->grid(ii,jj,inPixelIdx)*
				tEntry;
			else if (ifi)
			    ofi->grid(i,j,k)+=yfi->grid(ii,jj,inPixelIdx)*
				tEntry;
			else if (ifs)
			    ofs->grid(i,j,k)+=yfs->grid(ii,jj,inPixelIdx)*
				tEntry;
			else if (ifu)
			    ofu->grid(i,j,k)+=yfu->grid(ii,jj,inPixelIdx)*
				tEntry;
			else 
			    ofc->grid(i,j,k)+=yfc->grid(ii,jj,inPixelIdx)*
				tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(osf->nz, 2);
	buildOversampleTriangleTable(&table, osf->nz, Zratio);
	for (int k=0; k<osf->nz; k++) {
	    int left=floor(k/Zratio)+0;
	    int right=ceil(k/Zratio)+0;
	    double lEntry=table(k,0);
	    double rEntry=table(k,1);
	    for (int i=0, ii=0; i<osf->nx; i++, ii++)
		for (int j=0, jj=0; j<osf->ny; j++, jj++)
		    if (ifd)
			ofd->grid(i,j,k)=yfd->grid(ii,jj,left)*lEntry+
			    yfd->grid(ii,jj,right)*rEntry;
		    else if (iff)
			off->grid(i,j,k)=yff->grid(ii,jj,left)*lEntry+
			    yff->grid(ii,jj,right)*rEntry;
		    else if (ifi)
			ofi->grid(i,j,k)=yfi->grid(ii,jj,left)*lEntry+
			    yfi->grid(ii,jj,right)*rEntry;
		    else if (ifs)
			ofs->grid(i,j,k)=yfs->grid(ii,jj,left)*lEntry+
			    yfs->grid(ii,jj,right)*rEntry;
		    else if (ifu)
			ofu->grid(i,j,k)=yfu->grid(ii,jj,left)*lEntry+
			    yfu->grid(ii,jj,right)*rEntry;
		    else 
			ofc->grid(i,j,k)=yfc->grid(ii,jj,left)*lEntry+
			    yfc->grid(ii,jj,right)*rEntry;
	}
    }
}

void voteFilter() {
    ScalarFieldRGdouble *ifd, *xfd, *yfd, *ofd;
    ScalarFieldRGfloat *iff, *xff, *yff, *off;
    ScalarFieldRGint *ifi, *xfi, *yfi, *ofi;
    ScalarFieldRGshort *ifs, *xfs, *yfs, *ofs;
    ScalarFieldRGchar *ifc, *xfc, *yfc, *ofc;
    ScalarFieldRGuchar *ifu, *xfu, *yfu, *ofu;
    ifd=isf->getRGDouble();
    xfd=fldX->getRGDouble();
    yfd=fldY->getRGDouble();
    ofd=osf->getRGDouble();
    iff=isf->getRGFloat();
    xff=fldX->getRGFloat();
    yff=fldY->getRGFloat();
    off=osf->getRGFloat();
    ifi=isf->getRGInt();
    xfi=fldX->getRGInt();
    yfi=fldY->getRGInt();
    ofi=osf->getRGInt();
    ifs=isf->getRGShort();
    xfs=fldX->getRGShort();
    yfs=fldY->getRGShort();
    ofs=osf->getRGShort();
    ifc=isf->getRGChar();
    xfc=fldX->getRGChar();
    yfc=fldY->getRGChar();
    ofc=osf->getRGChar();
    ifu=isf->getRGUchar();
    xfu=fldX->getRGUchar();
    yfu=fldY->getRGUchar();
    ofu=osf->getRGUchar();
    
    double Xratio=(osf->nx-1.)/(isf->nx-1.);
    if (Xratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<fldX->nx; i++, ii++)
	    for (int j=0, jj=0; j<fldX->ny; j++, jj++)
		for (int k=0, kk=0; k<fldX->nz; k++, kk++)
		    if (ifd)
			xfd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			xff->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			xfi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else if (ifs)
			xfs->grid(i,j,k)=ifs->grid(ii, jj, kk);
		    else if (ifu)
			xfu->grid(i,j,k)=ifu->grid(ii, jj, kk);
		    else 
			xfc->grid(i,j,k)=ifc->grid(ii, jj, kk);
    } else if (Xratio<1) {		// undersampling     big->small
	int span=ceil(2./Xratio);
	if (ifs || ifi || ifc || ifu) {
	    double min_elem, max_elem;
	    if (ifi) ifi->get_minmax(min_elem, max_elem);
	    else if (ifs) ifs->get_minmax(min_elem, max_elem);
	    else if (ifu) ifu->get_minmax(min_elem, max_elem);
	    else ifc->get_minmax(min_elem, max_elem);
	    Array1<int> table((int)(max_elem-min_elem+1));
	    for (int i=0; i<fldX->nx; i++) {
		for (int j=0, jj=0; j<fldX->ny; j++, jj++)
		    for (int k=0, kk=0; k<fldX->nz; k++, kk++) {
			int l;
			for (l=0; l<table.size(); l++) table[l]=0;
			for (l=0; l<span; l++) {
			    int inPixelIdx=(int)((i-1)/Xratio+0+l);
			    if (inPixelIdx<0) inPixelIdx=0;
			    else if(inPixelIdx>isf->nx-1) inPixelIdx=isf->nx-1;
			    if (ifi)
				table[ifi->grid(inPixelIdx,jj,kk)-min_elem]++;
			    else if (ifs) 
				table[ifs->grid(inPixelIdx,jj,kk)-min_elem]++;
			    else if (ifu) 
				table[ifu->grid(inPixelIdx,jj,kk)-min_elem]++;
			    else 
				table[ifc->grid(inPixelIdx,jj,kk)-min_elem]++;
			}
			int max_idx=0;
			int max_vote=table[0];
			for (l=1; l<table.size(); l++) {
			    if (table[l] > max_vote) {
				max_idx=l;
				max_vote=table[l];
			    }
			}
			if (ifi) xfi->grid(i,j,k)=(int)(max_idx+min_elem);
			else if (ifu) 
			    xfu->grid(i,j,k)=(uchar)(max_idx+min_elem);
			else if (ifs) 
			    xfs->grid(i,j,k)=(short)(max_idx+min_elem);
			else xfc->grid(i,j,k)=(char)(max_idx+min_elem);
		    }
	    }
	} else {
	    Array2<double> table(fldX->nx, span);
	    buildUndersampleTriangleTable(&table, fldX->nx, span, Xratio);
	    for (int i=0; i<fldX->nx; i++) {
		for (int l=0; l<span; l++) {
		    double tEntry=table(i,l);
		    int inPixelIdx=(int)((i-1)/Xratio+0+l);
		    if (inPixelIdx<0) inPixelIdx=0;
		    else if (inPixelIdx>isf->nx-1) inPixelIdx=isf->nx-1;
		    for (int j=0, jj=0; j<fldX->ny; j++, jj++)
			for (int k=0, kk=0; k<fldX->nz; k++, kk++) 
			    if (ifd)
				xfd->grid(i,j,k)+=ifd->grid(inPixelIdx,jj,kk)*
				    tEntry;
			    else
				xff->grid(i,j,k)+=iff->grid(inPixelIdx,jj,kk)*
				    tEntry;
		}
	    }
	}
    } else {			// oversampling      small->big (just copy)
	double curr=0;
	for (int i=0; i<fldX->nx; i++, curr+=1./Xratio) {
	    for (int j=0, jj=0; j<fldX->ny; j++, jj++) {
		for (int k=0, kk=0; k<fldX->nz; k++, kk++) {
		    if (ifd)
			xfd->grid(i,j,k)=ifd->grid((int)curr+0,jj,kk);
		    else if (iff)
			xff->grid(i,j,k)=iff->grid((int)curr+0,jj,kk);
		    else if (ifi)
			xfi->grid(i,j,k)=ifi->grid((int)curr+0,jj,kk);
		    else if (ifs)
			xfs->grid(i,j,k)=ifs->grid((int)curr+0,jj,kk);
		    else if (ifu)
			xfu->grid(i,j,k)=ifu->grid((int)curr+0,jj,kk);
		    else 
			xfc->grid(i,j,k)=ifc->grid((int)curr+0,jj,kk);
		}
	    }
	}
    }
    double Yratio=(fldY->ny-1.)/(isf->ny-1.);
    if (Yratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<fldY->nx; i++, ii++)
	    for (int j=0, jj=0; j<fldY->ny; j++, jj++)
		for (int k=0, kk=0; k<fldY->nz; k++, kk++)
		    if (ifd)
			yfd->grid(i,j,k)=xfd->grid(ii, jj, kk);
		    else if (iff)
			yff->grid(i,j,k)=xff->grid(ii, jj, kk);
		    else if (ifi)
			yfi->grid(i,j,k)=xfi->grid(ii, jj, kk);
		    else if (ifs)
			yfs->grid(i,j,k)=xfs->grid(ii, jj, kk);
		    else if (ifu)
			yfu->grid(i,j,k)=xfu->grid(ii, jj, kk);
		    else 
			yfc->grid(i,j,k)=xfc->grid(ii, jj, kk);
    } else if (Yratio<1) {		// undersampling     big->small
	int span=ceil(2./Yratio);
	if (ifs || ifi || ifc || ifu) {
	    double min_elem, max_elem;
	    if (ifi) ifi->get_minmax(min_elem, max_elem);
	    else if (ifs) ifs->get_minmax(min_elem, max_elem);
	    else if (ifu) ifu->get_minmax(min_elem, max_elem);
	    else ifc->get_minmax(min_elem, max_elem);
	    Array1<int> table((int)(max_elem-min_elem+1));
	    for (int j=0; j<fldY->ny; j++) {
		for (int i=0, ii=0; i<fldY->nx; i++, ii++)
		    for (int k=0, kk=0; k<fldY->nz; k++, kk++) {
			int l;
			for (l=0; l<table.size(); l++) table[l]=0;
			for (l=0; l<span; l++) {
			    int inPixelIdx=(int)((j-1)/Yratio+0+l);
			    if (inPixelIdx<0) inPixelIdx=0;
			    else if(inPixelIdx>isf->ny-1) inPixelIdx=isf->ny-1;
			    if (ifi)
				table[xfi->grid(ii,inPixelIdx,kk)-min_elem]++;
			    else if (ifu)
				table[xfu->grid(ii,inPixelIdx,kk)-min_elem]++;
			    else if (ifs)
				table[xfs->grid(ii,inPixelIdx,kk)-min_elem]++;
			    else 
				table[xfc->grid(ii,inPixelIdx,kk)-min_elem]++;
			}
			int max_idx=0;
			int max_vote=table[0];
			for (l=1; l<table.size(); l++) {
			    if (table[l] > max_vote) {
				max_idx=l;
				max_vote=table[l];
			    }
			}
			if (ifi) yfi->grid(i,j,k)=(int)(max_idx+min_elem);
			else if (ifu) 
			    yfu->grid(i,j,k)=(uchar)(max_idx+min_elem);
			else if (ifs) 
			    yfs->grid(i,j,k)=(short)(max_idx+min_elem);
			else yfc->grid(i,j,k)=(char)(max_idx+min_elem);
		    }			
	    }
	} else {
	    Array2<double> table(fldY->ny, span);
	    buildUndersampleTriangleTable(&table, fldY->ny, span, Yratio);
	    for (int j=0; j<fldY->ny; j++) {
		for (int l=0; l<span; l++) {
		    double tEntry=table(j,l);
		    int inPixelIdx=(int)((j-1)/Yratio+0+l);
		    if (inPixelIdx<0) inPixelIdx=0;
		    else if (inPixelIdx>isf->ny-1) inPixelIdx=isf->ny-1;
		    for (int i=0, ii=0; i<fldY->nx; i++, ii++)
			for (int k=0, kk=0; k<fldY->nz; k++, kk++) 
			    if (ifd)
				yfd->grid(i,j,k)+=xfd->grid(ii,inPixelIdx,kk)*
				    tEntry;
			    else 
				yff->grid(i,j,k)+=xff->grid(ii,inPixelIdx,kk)*
				    tEntry;
		}
	    }
	}
    } else {			// oversampling      small->big  (just copy)
	double curr=0;
	for (int j=0; j<fldY->ny; j++, curr+=1./Yratio) {
	    for (int i=0, ii=0; i<fldY->nx; i++, ii++) {
		for (int k=0, kk=0; k<fldY->nz; k++, kk++) {
		    if (ifd)
			yfd->grid(i,j,k)=xfd->grid(ii,(int)curr+0,kk);
		    else if (iff)
			yff->grid(i,j,k)=xff->grid(ii,(int)curr+0,kk);
		    else if (ifi)
			yfi->grid(i,j,k)=xfi->grid(ii,(int)curr+0,kk);
		    else if (ifs)
			yfs->grid(i,j,k)=xfs->grid(ii,(int)curr+0,kk);
		    else if (ifu)
			yfu->grid(i,j,k)=xfu->grid(ii,(int)curr+0,kk);
		    else 
			yfc->grid(i,j,k)=xfc->grid(ii,(int)curr+0,kk);
		}
	    }
	}
    }
    double Zratio=(osf->nz-1.)/(isf->nz-1.);
    if (Zratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<osf->nx; i++, ii++)
	    for (int j=0, jj=0; j<osf->ny; j++, jj++)
		for (int k=0, kk=0; k<osf->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=yfd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=yff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=yfi->grid(ii, jj, kk);
		    else if (ifs)
			ofs->grid(i,j,k)=yfs->grid(ii, jj, kk);
		    else if (ifu)
			ofu->grid(i,j,k)=yfu->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=yfc->grid(ii, jj, kk);
    } else if (Zratio<1) {		// undersampling     big->small
	int span=ceil(2./Zratio);
	if (ifs || ifi || ifc || ifu) {
	    double min_elem, max_elem;
	    if (ifi) ifi->get_minmax(min_elem, max_elem);
	    else if (ifu) ifu->get_minmax(min_elem, max_elem);
	    else if (ifs) ifs->get_minmax(min_elem, max_elem);
	    else ifc->get_minmax(min_elem, max_elem);
	    Array1<int> table((int)(max_elem-min_elem+1));
	    for (int k=0; k<osf->nz; k++) {
		for (int i=0, ii=0; i<osf->nx; i++, ii++) 
		    for (int j=0, jj=0; j<osf->ny; j++, jj++) {
			int l;
			for (l=0; l<table.size(); l++) table[l]=0;
			for (l=0; l<span; l++) {
			    int inPixelIdx=(int)((k-1)/Zratio+0+l);
			    if (inPixelIdx<0) inPixelIdx=0;
			    else if(inPixelIdx>isf->nz-1) inPixelIdx=isf->nz-1;
			    if (ifi)
				table[yfi->grid(ii,jj,inPixelIdx)-min_elem]++;
			    else if (ifu)
				table[yfu->grid(ii,jj,inPixelIdx)-min_elem]++;
			    else if (ifs)
				table[yfs->grid(ii,jj,inPixelIdx)-min_elem]++;
			    else
				table[yfc->grid(ii,jj,inPixelIdx)-min_elem]++;
			}
			int max_idx=0;
			int max_vote=table[0];
			for (l=1; l<table.size(); l++) {
			    if (table[l] > max_vote) {
				max_idx=l;
				max_vote=table[l];
			    }
			}
			if (ifi) ofi->grid(i,j,k)=(int)(max_idx+min_elem);
			else if (ifu) 
			    ofu->grid(i,j,k)=(uchar)(max_idx+min_elem);
			else if (ifs) 
			    ofs->grid(i,j,k)=(short)(max_idx+min_elem);
			else ofc->grid(i,j,k)=(char)(max_idx+min_elem);
		    }
	    }	    
	} else {
	    Array2<double> table(osf->nz, span);
	    buildUndersampleTriangleTable(&table, osf->nz, span, Zratio);
	    for (int k=0; k<osf->nz; k++) {
		for (int l=0; l<span; l++) {
		    double tEntry=table(k,l);
		    int inPixelIdx=(int)((k-1)/Zratio+0+l);
		    if (inPixelIdx<0) inPixelIdx=0;
		    else if (inPixelIdx>isf->nz-1) inPixelIdx=isf->nz-1;
		    for (int i=0, ii=0; i<osf->nx; i++, ii++) 
			for (int j=0, jj=0; j<osf->ny; j++, jj++)
			    if (ifd)
				ofd->grid(i,j,k)+=yfd->grid(ii,jj,inPixelIdx)*
				    tEntry;
			    else
				off->grid(i,j,k)+=yff->grid(ii,jj,inPixelIdx)*
				    tEntry;
		}
	    }
	}
    } else {			// oversampling      small->big  (just copy)
	double curr=0;
	for (int k=0; k<osf->nz; k++, curr+=1./Zratio) {
	    for (int i=0, ii=0; i<osf->nx; i++, ii++) {
		for (int j=0, jj=0; j<osf->ny; j++, jj++) {
		    if (ifd)
			ofd->grid(i,j,k)=yfd->grid(ii,jj,(int)curr+0);
		    else if (iff)
			off->grid(i,j,k)=yff->grid(ii,jj,(int)curr+0);
		    else if (ifi)
			ofi->grid(i,j,k)=yfi->grid(ii,jj,(int)curr+0);
		    else if (ifs)
			ofs->grid(i,j,k)=yfs->grid(ii,jj,(int)curr+0);
		    else if (ifu)
			ofu->grid(i,j,k)=yfu->grid(ii,jj,(int)curr+0);
		    else 
			ofc->grid(i,j,k)=yfc->grid(ii,jj,(int)curr+0);
		}	
	    }
	}
    }
}

void voteFilterWeighted() {
    ScalarFieldRGdouble *ifd, *xfd, *yfd, *ofd;
    ScalarFieldRGfloat *iff, *xff, *yff, *off;
    ScalarFieldRGint *ifi, *xfi, *yfi, *ofi;
    ScalarFieldRGshort *ifs, *xfs, *yfs, *ofs;
    ScalarFieldRGchar *ifc, *xfc, *yfc, *ofc;
    ScalarFieldRGuchar *ifu, *xfu, *yfu, *ofu;
    ifd=isf->getRGDouble();
    xfd=fldX->getRGDouble();
    yfd=fldY->getRGDouble();
    ofd=osf->getRGDouble();
    iff=isf->getRGFloat();
    xff=fldX->getRGFloat();
    yff=fldY->getRGFloat();
    off=osf->getRGFloat();
    ifi=isf->getRGInt();
    xfi=fldX->getRGInt();
    yfi=fldY->getRGInt();
    ofi=osf->getRGInt();
    ifc=isf->getRGChar();
    xfc=fldX->getRGChar();
    yfc=fldY->getRGChar();
    ofc=osf->getRGChar();
    ifu=isf->getRGUchar();
    xfu=fldX->getRGUchar();
    yfu=fldY->getRGUchar();
    ofu=osf->getRGUchar();
    ifs=isf->getRGShort();
    xfs=fldX->getRGShort();
    yfs=fldY->getRGShort();
    ofs=osf->getRGShort();

    Array1<int> weights;
    weights.resize(6);
    weights[0]=1; weights[1]=4; weights[2]=4; weights[3]=1; weights[4]=1; weights[5]=1;
    double Xratio=(osf->nx-1.)/(isf->nx-1.);
    if (Xratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<fldX->nx; i++, ii++)
	    for (int j=0, jj=0; j<fldX->ny; j++, jj++)
		for (int k=0, kk=0; k<fldX->nz; k++, kk++)
		    if (ifd)
			xfd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			xff->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			xfi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else if (ifu)
			xfu->grid(i,j,k)=ifu->grid(ii, jj, kk);
		    else if (ifs)
			xfs->grid(i,j,k)=ifs->grid(ii, jj, kk);
		    else 
			xfc->grid(i,j,k)=ifc->grid(ii, jj, kk);
    } else if (Xratio<1) {		// undersampling     big->small
	int span=ceil(2./Xratio);
	if (ifi || ifc || ifu || ifs) {
	    double min_elem, max_elem;
	    if (ifi) ifi->get_minmax(min_elem, max_elem);
	    else if (ifu) ifu->get_minmax(min_elem, max_elem);
	    else if (ifs) ifs->get_minmax(min_elem, max_elem);
	    else ifc->get_minmax(min_elem, max_elem);
	    Array1<int> table((int)(max_elem-min_elem+1));
	    for (int i=0; i<fldX->nx; i++) {
		for (int j=0, jj=0; j<fldX->ny; j++, jj++)
		    for (int k=0, kk=0; k<fldX->nz; k++, kk++) {
			int l;
			for (l=0; l<table.size(); l++) table[l]=0;
			for (l=0; l<span; l++) {
			    int inPixelIdx=(int)((i-1)/Xratio+0+l);
			    if (inPixelIdx<0) inPixelIdx=0;
			    else if(inPixelIdx>isf->nx-1) inPixelIdx=isf->nx-1;
			    if (ifi)
				table[ifi->grid(inPixelIdx,jj,kk)-min_elem]++;
			    else if (ifu) 
				table[ifu->grid(inPixelIdx,jj,kk)-min_elem]++;
			    else if (ifs) 
				table[ifs->grid(inPixelIdx,jj,kk)-min_elem]++;
			    else 
				table[ifc->grid(inPixelIdx,jj,kk)-min_elem]++;
			}
			int max_idx=0;
			int max_vote=table[0];
			for (l=1; l<table.size(); l++) {
			    if (table[l]*weights[l] > max_vote) {
				max_idx=l;
				max_vote=table[l]*weights[l];
			    }
			}
			if (ifi) xfi->grid(i,j,k)=(int)(max_idx+min_elem);
			else if (ifu) 
			    xfu->grid(i,j,k)=(uchar)(max_idx+min_elem);
			else if (ifs) 
			    xfs->grid(i,j,k)=(short)(max_idx+min_elem);
			else xfc->grid(i,j,k)=(char)(max_idx+min_elem);
		    }
	    }
	} else {
	    Array2<double> table(fldX->nx, span);
	    buildUndersampleTriangleTable(&table, fldX->nx, span, Xratio);
	    for (int i=0; i<fldX->nx; i++) {
		for (int l=0; l<span; l++) {
		    double tEntry=table(i,l);
		    int inPixelIdx=(int)((i-1)/Xratio+0+l);
		    if (inPixelIdx<0) inPixelIdx=0;
		    else if (inPixelIdx>isf->nx-1) inPixelIdx=isf->nx-1;
		    for (int j=0, jj=0; j<fldX->ny; j++, jj++)
			for (int k=0, kk=0; k<fldX->nz; k++, kk++) 
			    if (ifd)
				xfd->grid(i,j,k)+=ifd->grid(inPixelIdx,jj,kk)*
				    tEntry;
			    else
				xff->grid(i,j,k)+=iff->grid(inPixelIdx,jj,kk)*
				    tEntry;
		}
	    }
	}
    } else {			// oversampling      small->big (just copy)
	double curr=0;
	for (int i=0; i<fldX->nx; i++, curr+=1./Xratio) {
	    for (int j=0, jj=0; j<fldX->ny; j++, jj++) {
		for (int k=0, kk=0; k<fldX->nz; k++, kk++) {
		    if (ifd)
			xfd->grid(i,j,k)=ifd->grid((int)curr+0,jj,kk);
		    else if (iff)
			xff->grid(i,j,k)=iff->grid((int)curr+0,jj,kk);
		    else if (ifi)
			xfi->grid(i,j,k)=ifi->grid((int)curr+0,jj,kk);
		    else if (ifu)
			xfu->grid(i,j,k)=ifu->grid((int)curr+0,jj,kk);
		    else if (ifs)
			xfs->grid(i,j,k)=ifs->grid((int)curr+0,jj,kk);
		    else 
			xfc->grid(i,j,k)=ifc->grid((int)curr+0,jj,kk);
		}
	    }
	}
    }
    double Yratio=(fldY->ny-1.)/(isf->ny-1.);
    if (Yratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<fldY->nx; i++, ii++)
	    for (int j=0, jj=0; j<fldY->ny; j++, jj++)
		for (int k=0, kk=0; k<fldY->nz; k++, kk++)
		    if (ifd)
			yfd->grid(i,j,k)=xfd->grid(ii, jj, kk);
		    else if (iff)
			yff->grid(i,j,k)=xff->grid(ii, jj, kk);
		    else if (ifi)
			yfi->grid(i,j,k)=xfi->grid(ii, jj, kk);
		    else if (ifu)
			yfu->grid(i,j,k)=xfu->grid(ii, jj, kk);
		    else if (ifs)
			yfs->grid(i,j,k)=xfs->grid(ii, jj, kk);
		    else 
			yfc->grid(i,j,k)=xfc->grid(ii, jj, kk);
    } else if (Yratio<1) {		// undersampling     big->small
	int span=ceil(2./Yratio);
	if (ifi || ifc || ifu || ifs) {
	    double min_elem, max_elem;
	    if (ifi) ifi->get_minmax(min_elem, max_elem);
	    else if (ifu) ifu->get_minmax(min_elem, max_elem);
	    else ifc->get_minmax(min_elem, max_elem);
	    Array1<int> table((int)(max_elem-min_elem+1));
	    for (int j=0; j<fldY->ny; j++) {
		for (int i=0, ii=0; i<fldY->nx; i++, ii++)
		    for (int k=0, kk=0; k<fldY->nz; k++, kk++) {
			int l;
			for (l=0; l<table.size(); l++) table[l]=0;
			for (l=0; l<span; l++) {
			    int inPixelIdx=(int)((j-1)/Yratio+0+l);
			    if (inPixelIdx<0) inPixelIdx=0;
			    else if(inPixelIdx>isf->ny-1) inPixelIdx=isf->ny-1;
			    if (ifi)
				table[xfi->grid(ii,inPixelIdx,kk)-min_elem]++;
			    else if (ifu)
				table[xfu->grid(ii,inPixelIdx,kk)-min_elem]++;
			    else if (ifs)
				table[xfs->grid(ii,inPixelIdx,kk)-min_elem]++;
			    else 
				table[xfc->grid(ii,inPixelIdx,kk)-min_elem]++;
			}
			int max_idx=0;
			int max_vote=table[0];
			for (l=1; l<table.size(); l++) {
			    if (table[l]*weights[l] > max_vote) {
				max_idx=l;
				max_vote=table[l]*weights[l];
			    }
			}
			if (ifi) yfi->grid(i,j,k)=(int)(max_idx+min_elem);
			else if (ifu) 
			    yfu->grid(i,j,k)=(uchar)(max_idx+min_elem);
			else if (ifs) 
			    yfs->grid(i,j,k)=(short)(max_idx+min_elem);
			else yfc->grid(i,j,k)=(char)(max_idx+min_elem);
		    }			
	    }
	} else {
	    Array2<double> table(fldY->ny, span);
	    buildUndersampleTriangleTable(&table, fldY->ny, span, Yratio);
	    for (int j=0; j<fldY->ny; j++) {
		for (int l=0; l<span; l++) {
		    double tEntry=table(j,l);
		    int inPixelIdx=(int)((j-1)/Yratio+0+l);
		    if (inPixelIdx<0) inPixelIdx=0;
		    else if (inPixelIdx>isf->ny-1) inPixelIdx=isf->ny-1;
		    for (int i=0, ii=0; i<fldY->nx; i++, ii++)
			for (int k=0, kk=0; k<fldY->nz; k++, kk++) 
			    if (ifd)
				yfd->grid(i,j,k)+=xfd->grid(ii,inPixelIdx,kk)*
				    tEntry;
			    else 
				yff->grid(i,j,k)+=xff->grid(ii,inPixelIdx,kk)*
				    tEntry;
		}
	    }
	}
    } else {			// oversampling      small->big  (just copy)
	double curr=0;
	for (int j=0; j<fldY->ny; j++, curr+=1./Yratio) {
	    for (int i=0, ii=0; i<fldY->nx; i++, ii++) {
		for (int k=0, kk=0; k<fldY->nz; k++, kk++) {
		    if (ifd)
			yfd->grid(i,j,k)=xfd->grid(ii,(int)curr+0,kk);
		    else if (iff)
			yff->grid(i,j,k)=xff->grid(ii,(int)curr+0,kk);
		    else if (ifi)
			yfi->grid(i,j,k)=xfi->grid(ii,(int)curr+0,kk);
		    else if (ifu)
			yfu->grid(i,j,k)=xfu->grid(ii,(int)curr+0,kk);
		    else if (ifs)
			yfs->grid(i,j,k)=xfs->grid(ii,(int)curr+0,kk);
		    else 
			yfc->grid(i,j,k)=xfc->grid(ii,(int)curr+0,kk);
		}
	    }
	}
    }
    double Zratio=(osf->nz-1.)/(isf->nz-1.);
    if (Zratio == 1) {		// trivial filter
	for (int i=0, ii=0; i<osf->nx; i++, ii++)
	    for (int j=0, jj=0; j<osf->ny; j++, jj++)
		for (int k=0, kk=0; k<osf->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=yfd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=yff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=yfi->grid(ii, jj, kk);
		    else if (ifu)
			ofu->grid(i,j,k)=yfu->grid(ii, jj, kk);
		    else if (ifs)
			ofs->grid(i,j,k)=yfs->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=yfc->grid(ii, jj, kk);
    } else if (Zratio<1) {		// undersampling     big->small
	int span=ceil(2./Zratio);
	if (ifi || ifc || ifu || ifs) {
	    double min_elem, max_elem;
	    if (ifi) ifi->get_minmax(min_elem, max_elem);
	    else if (ifu) ifu->get_minmax(min_elem, max_elem);
	    else ifc->get_minmax(min_elem, max_elem);
	    Array1<int> table((int)(max_elem-min_elem+1));
	    for (int k=0; k<osf->nz; k++) {
		for (int i=0, ii=0; i<osf->nx; i++, ii++) 
		    for (int j=0, jj=0; j<osf->ny; j++, jj++) {
			int l;
			for (l=0; l<table.size(); l++) table[l]=0;
			for (l=0; l<span; l++) {
			    int inPixelIdx=(int)((k-1)/Zratio+0+l);
			    if (inPixelIdx<0) inPixelIdx=0;
			    else if(inPixelIdx>isf->nz-1) inPixelIdx=isf->nz-1;
			    if (ifi)
				table[yfi->grid(ii,jj,inPixelIdx)-min_elem]++;
			    else if (ifu)
				table[yfu->grid(ii,jj,inPixelIdx)-min_elem]++;
			    else if (ifs)
				table[yfs->grid(ii,jj,inPixelIdx)-min_elem]++;
			    else
				table[yfc->grid(ii,jj,inPixelIdx)-min_elem]++;
			}
			int max_idx=0;
			int max_vote=table[0];
			for (l=1; l<table.size(); l++) {
			    if (table[l]*weights[l] > max_vote) {
				max_idx=l;
				max_vote=table[l]*weights[l];
			    }
			}
			if (ifi) ofi->grid(i,j,k)=(int)(max_idx+min_elem);
			else if (ifu) 
			    ofu->grid(i,j,k)=(uchar)(max_idx+min_elem);
			else if (ifs) 
			    ofs->grid(i,j,k)=(short)(max_idx+min_elem);
			else ofc->grid(i,j,k)=(char)(max_idx+min_elem);
		    }
	    }	    
	} else {
	    Array2<double> table(osf->nz, span);
	    buildUndersampleTriangleTable(&table, osf->nz, span, Zratio);
	    for (int k=0; k<osf->nz; k++) {
		for (int l=0; l<span; l++) {
		    double tEntry=table(k,l);
		    int inPixelIdx=(int)((k-1)/Zratio+0+l);
		    if (inPixelIdx<0) inPixelIdx=0;
		    else if (inPixelIdx>isf->nz-1) inPixelIdx=isf->nz-1;
		    for (int i=0, ii=0; i<osf->nx; i++, ii++) 
			for (int j=0, jj=0; j<osf->ny; j++, jj++)
			    if (ifd)
				ofd->grid(i,j,k)+=yfd->grid(ii,jj,inPixelIdx)*
				    tEntry;
			    else
				off->grid(i,j,k)+=yff->grid(ii,jj,inPixelIdx)*
				    tEntry;
		}
	    }
	}
    } else {			// oversampling      small->big  (just copy)
	double curr=0;
	for (int k=0; k<osf->nz; k++, curr+=1./Zratio) {
	    for (int i=0, ii=0; i<osf->nx; i++, ii++) {
		for (int j=0, jj=0; j<osf->ny; j++, jj++) {
		    if (ifd)
			ofd->grid(i,j,k)=yfd->grid(ii,jj,(int)curr+0);
		    else if (iff)
			off->grid(i,j,k)=yff->grid(ii,jj,(int)curr+0);
		    else if (ifi)
			ofi->grid(i,j,k)=yfi->grid(ii,jj,(int)curr+0);
		    else if (ifu)
			ofu->grid(i,j,k)=yfu->grid(ii,jj,(int)curr+0);
		    else if (ifs)
			ofs->grid(i,j,k)=yfs->grid(ii,jj,(int)curr+0);
		    else 
			ofc->grid(i,j,k)=yfc->grid(ii,jj,(int)curr+0);
		}	
	    }
	}
    }
}

void printTable(Array2<double>*a) {
    cerr << "Filter Table (" << a->dim1() << "," << a->dim2() << ")\n";
    for (int i=0; i<a->dim1(); i++) {
	for (int j=0; j<a->dim2(); j++) {
	    cerr << (*a)(i,j) << " ";
	}
	cerr << "\n";
    }
    cerr << "\n";
}
  
void buildUndersampleTriangleTable(Array2<double> *table,
				   int size, int span, 
				   double ratio) {
    double invRatio=1./ratio;
    for (int i=0; i<size; i++) {
        double total=0;
	double inCtr=i*invRatio;
	int inIdx=inCtr-invRatio;
	int j;
	for (j=0; j<span; j++, inIdx++) {
	    double val=invRatio-fabs(inCtr-inIdx);
	    if (val<0) {
		val=0;
	    }
	    (*table)(i,j)=val;
	    total+=val;
	}
	for (j=0; j<span; j++) {
	    (*table)(i,j)/=total;
	}	
    }
//    printTable(table);
}

void buildOversampleTriangleTable(Array2<double> *table,
				  int size, double ratio) {
    double inverse=1./ratio;
    double curr=1.;
    for (int i=0; i<size; i++) {
	(*table)(i,0)=curr;
	(*table)(i,1)=1.-curr;
	curr-=inverse;
	if (curr<0) curr+=1.;
    }
}
