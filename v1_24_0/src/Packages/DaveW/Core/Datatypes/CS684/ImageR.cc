
/*
 *  Image.cc: The ImageXYZ and ImageRM datatypes - used in the Raytracer
 *	      and Radioisity code.  These types are derived from the
 *	      VoidStar class.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/CS684/ImageR.h>
#include <Packages/DaveW/Core/Datatypes/CS684/xyz.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/NotFinished.h>

#include <iostream>
using std::cerr;
#include <stdlib.h>

namespace DaveW {
using namespace SCIRun;


static Persistent* make_ImageXYZ()
{
    return scinew ImageXYZ;
}
PersistentTypeID ImageXYZ::type_id("ImageXYZ", "VoidStar", make_ImageXYZ);

ImageXYZ::ImageXYZ()
: VoidStar()
{
}

ImageXYZ::ImageXYZ(const Array2<double>& XYZ) : VoidStar()
{
    xyz=XYZ;
}

ImageXYZ::ImageXYZ(const ImageXYZ& copy)
: xyz(copy.xyz), VoidStar(copy)
{
    NOT_FINISHED("ImageXYZ::ImageXYZ");
}

ImageXYZ::~ImageXYZ() {
}

VoidStar* ImageXYZ::clone()
{
    return scinew ImageXYZ(*this);
}

static Persistent* make_ImageRM()
{
    return scinew ImageRM;
}
PersistentTypeID ImageRM::type_id("ImageRM", "VoidStar", make_ImageRM);

ImageRM::ImageRM()
: VoidStar()
{
}

ImageRM::ImageRM(const Array2<Pixel>& p, const Array1<Spectrum>& ls, 
	    const Array1<clString>& ln, const Array1<Spectrum>& ms,
	    const Array1<clString>& mn, const Array1<double>& d,
	    const Array1<double>& s, const Spectrum& a) : 
    VoidStar(), pix(p), lightSpec(ls), lightName(ln),
    matlSpec(ms), matlName(mn), kd(d), ks(s), ka(a), LS(0), MS(0), KAS(0)
{
}

ImageRM::ImageRM(const ImageRM& copy)
: pix(copy.pix), lightSpec(copy.lightSpec), lightName(copy.lightName),
  matlSpec(copy.matlSpec), matlName(copy.matlName), kd(copy.kd), 
  ks(copy.ks), ka(copy.ka), VoidStar(copy), LS(0), MS(0), KAS(0)
{
    NOT_FINISHED("ImageRM::ImageRM");
}

void ImageRM::getSpectraMinMax(double &min, double &max) {
    cerr << "ImageRM::getSpectraMinMax() NOT IMPLEMENTED!\n";
    min=400;
    max=1000;
}

void ImageRM::bldSpecLightMatrix(int lindex) {
    Array1<double> specvals(num);
    lightSpec[lindex].rediscretize(specvals, min, max);
    for (int j=0; j<num; j++) (*LS)[j][lindex]=specvals[j];
}

void ImageRM::bldSpecMaterialMatrix(int mindex) {
    Array1<double> specvals(num);
    matlSpec[mindex].rediscretize(specvals, min, max);
    for (int j=0; j<num; j++) (*MS)[mindex][j]=specvals[j];
}

void ImageRM::bldSpecAmbientMatrix() {
    Array1<double> specvals(num);
    ka.rediscretize(specvals, min, max);
    for (int j=0; j<num; j++) (*KAS)[j][0]=specvals[j];
}

void ImageRM::bldSpecMatrices() {

    // to build our Spectral matrices, we'll discretize one spectrum at a
    // time and then copy the discretized values into a row of the matrix

    if (MS) free(MS);
    MS = new DenseMatrix(matlSpec.size(), num);
    int i;
    for (i=0; i<matlSpec.size(); i++) {
	bldSpecMaterialMatrix(i);
    }

    if (LS) free(LS);
    LS = new DenseMatrix(num, lightSpec.size());
    for (i=0; i<lightSpec.size(); i++) {
	bldSpecLightMatrix(i);
    }

    if (KAS) free(KAS);
    KAS = new DenseMatrix(num, 1);
    bldSpecAmbientMatrix();
}

// R = B(l,1) kd + [B(l,2) kd + B(l,3) kd B(3,2) ks] B(2,1) ks
void ImageRM::bldPixelR() {
    if (pix.dim1() + pix.dim2() == 0) {
	cerr << "NO PIXELS!\n";
	return;
    }

    int nb, nm, nl;
    nb=pix(0,0).D.size();
    nm=matlSpec.size();
    nl=lightSpec.size();
    
    if (kd.size()!=nm || ks.size()!=nm) {
	cerr << "ERROR -- ks and kd matlSpec must be the same size!\n";
	return;
    }
    
    DenseMatrix KD(nm, nm);
    DenseMatrix KS(nm, nm);
    DenseMatrix S0M(nm, nm);
    KS.zero(); KD.zero();
    
    int i;
    for (i=0; i<nm; i++) {
//	KS[i][i]=ks[i];
	KS[i][i]=1;
	KD[i][i]=kd[i];
    }

    int j;
    for (j=0; j<pix.dim1(); j++) {
	for (i=0; i<pix.dim2(); i++) {
//	    cerr << "Building R for pixel ("<<j<<","<<i<<")\n";
	    if (pix(j,i).E.size() != nl) {
		cerr << "Error in pixel("<<j<<","<<i<<") bad E.\n";
		return;
	    }

	    // gotta check the S and M matrices...
	    int k;
	    for (k=1; k<nb; k++) {
		if (pix(j,i).S[k-1].nrows() != nm || 
		    pix(j,i).S[k-1].ncols() != nm) {
		    cerr <<"Error in pixel("<<j<<","<<i<<") bad S["<<k-1<<"\n";
		    return;
		}
//		cerr << "Here's S["<<k-1<<"] - ";
//		pix(j,i).S[k-1].print();
		if (pix(j,i).D[k].nrows() != nl || 
		    pix(j,i).D[k].ncols() != nm) {
		    cerr << "Error in pixel("<<j<<","<<i<<") bad D["<<k<<".\n";
		    return;
		}
//		cerr << "Here's D["<<k<<"] - \n";
//		pix(j,i).D[k].print();
	    }

	    // sizes look ok, let's build R...
	    DenseMatrix Rt(nl, nm);
	    DenseMatrix tmp1(nl, nm);
	    DenseMatrix tmp2(nl, nm);

	    // initialize Rt with the last bounce's direct, scaled by kd
	    Mult(Rt, pix(j,i).D[nb-1], KD);
	    cerr << "Pixel("<<j<<","<<i<<")...\n";
	    cerr << "Rt=";Rt.print();
	    // for each bounce, multiply by the specular, add the diffuse,
	    // save in Rt
	    for (k=nb-2; k>=0; k--) {
		Mult(tmp2, Rt, pix(j,i).S[k]);
	    cerr << "tmp2=";tmp2.print();
		Mult(tmp1, tmp2, KS);
	    cerr << "tmp1=";tmp1.print();
		Mult(tmp2, pix(j,i).D[k], KD);
	    cerr << "tmp2=";tmp2.print();
		Add(Rt, tmp1, tmp2);
	    cerr << "Rt=";Rt.print();
	    }

	    // gotta check the S0 vector...
	    if (pix(j,i).S0.size() != nm) {
		cerr << "Error in pixel("<<j<<","<<i<<") bad S0.\n";
		return;
	    }	    

#if 0
	    // S0 is the probability that a ray from the eye sees each matl
	    // Make it into a matrix, so we can multiply by it to get R
	    S0M.zero();
	    for (k=0; k<nm; k++) S0M[k][k]=pix(j,i).S0[k];
	    Mult(tmp1, Rt, S0M);
	    pix(j,i).R=tmp1;
#endif
	    pix(j,i).R=Rt;

//	    cerr << "Here's R: \n";
//	    pix(j,i).R.print();
	}
    }
}

// We have an R matrix for each pixel.  Compute the spectrum one lambda at a
// time for each pixel...
// for each wavelenght: multiply R by the material vector for that
// wavelength, add in the emitted light vector, multiply by the light vector
// for that wavelength, and add in the ambient value for that wavelength

void ImageRM::bldPixelSpectrum() {
    int l,w,x,y;
    int nl=lightSpec.size();
    int nm=matlSpec.size();
    int nw=KAS->nrows();
    int ny=pix.dim1();
    int nx=pix.dim2();
    
    DenseMatrix RM(nl,1);
    DenseMatrix ERM(nl,1);
    DenseMatrix LERM(1,1);
    DenseMatrix ALERM(1,1);
    DenseMatrix E(nl,1);
    DenseMatrix MSlambda(nm,1);
    DenseMatrix LSlambda(1,nl);
    
    for (y=0; y<ny; y++) {
	for (x=0; x<nx; x++) {
	    for (l=0; l<nl; l++) E[l][0]=pix(y,x).E[l];
	    double *sp=new double[nw];
	    for (w=0; w<nw; w++) {			// for each wavelength
		for (int mm=0; mm<nm; mm++) MSlambda[mm][0]=(*MS)[mm][w];
		for (int ll=0; ll<nl; ll++) LSlambda[0][ll]=(*LS)[w][ll];
		Mult(RM, pix(y,x).R, MSlambda);
		Add(ERM, E, RM);
		Mult(LERM, LSlambda, ERM);

		LERM[0][0] /= pix(y,x).nSamples;

		ALERM[0][0]=(*KAS)[w][0]+LERM[0][0];

//		Add(ALERM, *KAS, LERM);

		sp[w]=ALERM[0][0];
	    }
	    pix(y,x).spec=LiteSpectrum(min, max, num, spacing, sp);
//	    cerr << "Pixel ("<<y<<","<<x<<") spec = ";
//	    for (int i=0; i<nw; i++)
//		cerr << sp[i]<<" ";
//	    cerr << "\n";
	}
    }
}

void ImageRM::bldPixelXYZandRGB() {
    Spectrum XSpectra(xAmplitudes, xSpacing, xMinWavelength, xNumEntries);
    Spectrum YSpectra(yAmplitudes, ySpacing, yMinWavelength, yNumEntries);
    Spectrum ZSpectra(zAmplitudes, zSpacing, zMinWavelength, zNumEntries);

    Array1<double> tempXSpectra(num);
    Array1<double> tempYSpectra(num);
    Array1<double> tempZSpectra(num);

    XSpectra.rediscretize(tempXSpectra, min, max);
    YSpectra.rediscretize(tempYSpectra, min, max);
    ZSpectra.rediscretize(tempZSpectra, min, max);
    
    int x,y;
    int nx=pix.dim2();
    int ny=pix.dim1();

    for (y=0; y<ny; y++) {
	for (x=0; x<nx; x++) {
	    Point xyz(pix(y,x).spec.xyz(&(tempXSpectra[0]), 
					&(tempYSpectra[0]), 
					&(tempZSpectra[0])));
//	    cerr << "Pixel ("<<y<<","<<x<<") xyz = "<<xyz<<"\n";
	    Color clr=XYZ_to_RGB(xyz);
	    pix(y,x).xyz=xyz;
	    pix(y,x).c.red=Min(255.,Max(clr.r()*255.,0.));
	    pix(y,x).c.green=Min(255.,Max(clr.g()*255.,0.));
	    pix(y,x).c.blue=Min(255.,Max(clr.b()*255.,0.));
	}
    }
}

ImageRM::~ImageRM() {
}

VoidStar* ImageRM::clone()
{
    return scinew ImageRM(*this);
}

#define ImageXYZ_VERSION 1
void ImageXYZ::io(Piostream& stream) {

    /* int version=*/stream.begin_class("ImageXYZ", ImageXYZ_VERSION);
    VoidStar::io(stream);
    SCIRun::Pio(stream, xyz);
    stream.end_class();
}

#define ImageRM_VERSION 1
void ImageRM::io(SCIRun::Piostream& stream) {
  using SCIRun::Pio;
  using DaveW::Pio;
  /* int version=*/stream.begin_class("ImageRM", ImageRM_VERSION);
  VoidStar::io(stream);
  Pio(stream, pix);
  Pio(stream, lightSpec);
  Pio(stream, lightName);
  Pio(stream, matlSpec);
  Pio(stream, matlName);
  Pio(stream, kd);
  Pio(stream, ks);
  Pio(stream, ka);
  Pio(stream, min);
  Pio(stream, max);
  Pio(stream, num);
  Pio(stream, spacing);
  stream.end_class();
} 

}// End namespace DaveW


