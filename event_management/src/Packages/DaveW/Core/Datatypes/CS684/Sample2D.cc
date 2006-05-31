
/*
 *  Sample2D.cc: Generate Sample2D points in a domain
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/CS684/Sample2D.h>
#include <Core/Containers/Array2.h>
#include <Core/Math/Expon.h>

#include <iostream>
using std::cerr;

namespace DaveW {
using namespace SCIRun;

Sample2D::Sample2D(int mr)
: mr(mr)
{
}

Sample2D::Sample2D(const Sample2D &copy)
: mr(copy.mr)
{
}

Sample2D::~Sample2D() {
}

clString Sample2D::getMethod() const {
    if (rep==Jittered) return "Jittered";
    else if (rep==Random) return "Random";
    else if (rep==Regular) return "Regular";
    else if (rep==Poisson) return "Poisson";
    else if (rep==MultiJittered) return "Multijittered";
    else if (rep==QuasiMonteCarlo) return "QuasiMonteCarlo";
    else return ("unknown");
}

void Sample2D::setMethod(const clString& m) {
    if (m=="Jittered") rep=Jittered;
    else if (m=="Random") rep=Random;
    else if (m=="Regular") rep=Regular;
    else if (m=="Poisson") rep=Poisson;
    else if (m=="MultiJittered") rep=MultiJittered;
    else if (m=="QuasiMonteCarlo") rep=QuasiMonteCarlo;
    else cerr << "Sample2D error:  don't recognize method: "<<m<<"\n";
}

void Sample2D::genSamples(Array1<double>&x, Array1<double>&y, 
			  Array1<double>&w, int ns) {
    if (x.size() != y.size() || x.size() != w.size() || x.size() != ns) {
	x.resize(ns);
	y.resize(ns);
	w.resize(ns);
    }
    if (rep==Jittered) genSamplesJittered(x,y,w);
    else if (rep==Random) genSamplesRandom(x,y,w);
    else if (rep==Regular) genSamplesRegular(x,y,w);
    else if (rep==Poisson) genSamplesPoisson(x,y,w);
    else if (rep==MultiJittered) genSamplesMultiJittered(x,y,w);
    else cerr << "Sample2D::genSamples error -- can't generate samples.\n";
}

void Sample2D::genSamplesJittered(Array1<double>&x, Array1<double>&y,
				  Array1<double>&w) {
    double ww=1./(x.size());
    int ns=Sqrt(x.size());
    if (ns*ns != x.size()) {
	cerr << "ERROR:  need a perfect square number of jittered samples!\n";
    }
    double d=1./ns;
    double yv=0;
    int cnt=0;
    for (int j=0; j<ns; j++) {
	double xv=0;
	for (int i=0; i<ns; i++, cnt++) {
	    x[cnt]=xv+mr()*d;
	    y[cnt]=yv+mr()*d;
	    w[cnt]=ww;
	    xv+=d;
	}
	yv+=d;
    }
}

void Sample2D::genSamplesRandom(Array1<double>&x, Array1<double>&y,
				Array1<double>&w) {
    double ww=1./(x.size());
    for (int s=0; s<x.size(); s++) {
	x[s]=mr();
	y[s]=mr();
	w[s]=ww;
    }
}

void Sample2D::genSamplesRegular(Array1<double>&x, Array1<double>&y,
				 Array1<double>&w) {
    double ww=1./(x.size());
    int ns=Sqrt(x.size());
    double d=1./ns;
    double yv=d/2;
    int cnt=0;
    for (int j=0; j<ns; j++) {
	double xv=d/2;
	for (int i=0; i<ns; i++, cnt++) {
	    x[cnt]=xv;
	    y[cnt]=yv;
	    w[cnt]=ww;
	    xv+=d;
	}
	yv+=d;
    }
}

void Sample2D::genSamplesPoisson(Array1<double>&x, Array1<double>&y,
				 Array1<double>&w) {
    double ww=1./(x.size());
    double rr=ww*ww/4;
    for (int s=0; s<x.size(); s++) {
	int bad;
	do {
	    bad=0;
	    double x0=x[s]=mr();
	    double y0=y[s]=mr();
	    for (int i=0; i<s; i++) {
		double xx=x[i]-x0;
		double yy=y[i]-y0;
		if(xx*xx+yy*yy<rr) {
		    bad=1;
		    break;
		}
	    }
	} while(bad);
	w[s]=ww;
    }    
}

void Sample2D::genSamplesMultiJittered(Array1<double>&x, Array1<double>&y,
				       Array1<double>&w) {
    double ww=1./(x.size());
    int ns=x.size();
    int n_blocks=Sqrt(ns*1.); 

    Array1<int> node(ns);
    Array2<int> row_filled(n_blocks,n_blocks); // which row lines of which 
    row_filled.initialize(0); 		       // blocks are filled

    for (int col_block=0; col_block<n_blocks; col_block++) {

	// build an array of randomly ordered indices from 0..n_blocks
	Array1<int> col_line_idx(n_blocks);
	int i;
	for (i=0; i<n_blocks; i++) col_line_idx[i]=-1;
	for (i=0; i<n_blocks; i++) {
	    int rndm=mr()*(n_blocks-i);
	    int e=0;
	    for (; rndm>=0; e++) {
		if (col_line_idx[e]==-1) rndm--;
	    }
	    e--;
	    col_line_idx[e]=i;
	}

	// for each line in this block find a vacant row
	for (int col_line=0; col_line<n_blocks; col_line++) {
	    int row_block=col_line_idx[col_line];
	    
	    // find a vacant row in this block
	    int entry=mr()*(n_blocks-col_block);
	    int e=0;
	    for (; entry>=0; e++) {
		if (!row_filled(row_block,e)) entry--;
	    }
	    e--;
	    row_filled(row_block,e)=1;
	    node[col_block*n_blocks+col_line]=row_block*n_blocks+e;
	}
    }

    for (int n=0; n<node.size(); n++) {
	x[n]=(n+mr())*ww;
	y[n]=(node[n]+mr())*ww;
	w[n]=ww;
    }
}

void Sample2D::genSamplesQuasiMonteCarlo(Array1<double>&x, Array1<double>&y,
					 int start, int stop, int max) {
    double p, u;
    int k, kk;

    for (k=start; k<stop; k++) {
	u=0;
	for (p=0.5, kk=k; kk; p*=0.5, kk>>=1)
	    if (kk & 1) u+=p;
	x[k-start]=u;
	y[k-start]=(k+0.5)/max;
    }
}
} // End namespace DaveW

