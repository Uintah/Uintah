
/*
 *  DRaytracer.cc:  Project parallel rays at a sphere and see where they go
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/CS684/Pixel.h>
#include <Packages/DaveW/Core/Datatypes/CS684/Spectrum.h>
#include <Packages/DaveW/Core/Datatypes/CS684/DRaytracer.h>
#include <Packages/DaveW/Core/Datatypes/CS684/xyz.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using std::cerr;

namespace DaveW {
using namespace SCIRun;

int global_numbounces;
Mutex global_bounces_mutex("global_bounces_mutex");

static Persistent* make_DRaytracer()
{
  return scinew DRaytracer;
}

PersistentTypeID DRaytracer::type_id("DRaytracer", "VoidStar", make_DRaytracer);

DRaytracer::DRaytracer()
  : VoidStar()
{
  camera.fLength=0.05;
  camera.zoom=1;
  nx=100;
  ny=100;
  specMin=300.;
  specMax=700.;
  specNum=41;
}

DRaytracer::DRaytracer(const DRaytracer& copy)
  : VoidStar(copy), scene(copy.scene), camera(copy.camera), ns(copy.ns), 
  nx(copy.nx), ny(copy.ny), specMin(copy.specMin),
  specMax(copy.specMax), specNum(copy.specNum)
{
}

DRaytracer::~DRaytracer() {
}

VoidStar* DRaytracer::clone()
{
  return scinew DRaytracer(*this);
}

Color DRaytracer::spectrumToClr(Array1<double> &s) {
  LiteSpectrum l(specMin, specMax, specNum,
		 (specMax-specMin)/(specNum-1), &(s[0]));
  Point xyz(l.xyz(&(tempXSpectra[0]), &(tempYSpectra[0]), 
		  &(tempZSpectra[0])));
  Color clr=XYZ_to_RGB(xyz);
  return clr;
}

Point DRaytracer::spectrumToXYZ(Array1<double> &s) {
  LiteSpectrum l(specMin, specMax, specNum,
		 (specMax-specMin)/(specNum-1), &(s[0]));
  return l.xyz(&(tempXSpectra[0]), &(tempYSpectra[0]), 
	       &(tempZSpectra[0]));
}

Color DRaytracer::singleTrace(const Point& curr, Point& xyz, Pixel* p) {

  camera.initialize();

  Array1<double> lambda(specNum);
  lambda.initialize(0);
  //cerr << "ns="<<ns<<"\n";
  if (ns == 0) {    // shoot em all through the middle
    RTRay ray(camera.view.eyep(), curr, lambda, p);
    scene.trace(ray, 0);
  } else {	
    Array1<double> lambda_temp(specNum);
    lensPixelSamples(s, pixX, pixY, lensX, lensY, w, shuffle);
    for (int smpl=0; smpl<ns; smpl++) {
      Point from(camera.view.eyep()+(camera.u*lensX[smpl])+
		 (camera.v*lensY[smpl]));
      Point to(curr+(camera.u*(stepX*pixX[smpl]))+
	       (camera.v*(stepY*pixY[smpl])));
      RTRay r(from, to, lambda_temp, p);
      lambda_temp.initialize(0);
      scene.trace(r, 0);
      vectorAddScale(lambda, lambda_temp, 1);
    }
    vectorScaleBy(lambda, 1./ns);
  }
  //    cerr << "Lambda: ";
  //    for (int i=0; i<lambda.size(); i++) cerr << lambda[i]<<" ";

  //    cerr << "From singleTrace() spec = ";
  //    for (int i=0; i<specNum; i++) cerr << lambda[i]<<" ";
  //    cerr << "\n";

  LiteSpectrum l(specMin, specMax, specNum, (specMax-specMin)/(specNum-1),
		 &(lambda[0]));
  xyz=Point (l.xyz(&(tempXSpectra[0]), &(tempYSpectra[0]), 
		   &(tempZSpectra[0])));

  //    cerr <<"     xyz = "<<xyz<<"\n";
  Color clr=XYZ_to_RGB(xyz);
  //    cerr << "  clr="<<clr.r()<<","<<clr.g()<<","<<clr.b()<<"\n";
  return clr;
}

void DRaytracer::buildTempXYZSpectra() {
  Spectrum XSpectra(xAmplitudes, xSpacing, xMinWavelength, xNumEntries);
  Spectrum YSpectra(yAmplitudes, ySpacing, yMinWavelength, yNumEntries);
  Spectrum ZSpectra(zAmplitudes, zSpacing, zMinWavelength, zNumEntries);

  tempXSpectra.resize(specNum);
  tempYSpectra.resize(specNum);
  tempZSpectra.resize(specNum);

  XSpectra.rediscretize(tempXSpectra, specMin, specMax);
  YSpectra.rediscretize(tempYSpectra, specMin, specMax);
  ZSpectra.rediscretize(tempZSpectra, specMin, specMax);
}
    
void DRaytracer::preRayTrace() {
  global_numbounces=0;
  camera.initialize();
  double ratio, stepSize;

  ratio=nx*1./ny;
  if (ratio>1) {
    stepSize=2.*camera.fDist/(nx*camera.zoom);
  } else {
    stepSize=2.*camera.fDist/(ny*camera.zoom);
  }

  stepX=stepY=stepSize;	// should remove stepX and stepY from class!
  midPt=camera.view.eyep()+camera.w*camera.fDist;
  topLeft=midPt-Vector(camera.u*(stepSize*nx/2)+camera.v*(stepSize*ny/2));
  pixX.resize(ns);
  pixY.resize(ns);
  lensX.resize(ns);
  lensY.resize(ns);
  shuffle.resize(ns);
  s.setMethod("Jittered");
  scene.setupTempSpectra(specMin, specMax, specNum);
  buildTempXYZSpectra();


  int ii;
  for (ii=0; ii<scene.obj.size(); ii++)
    cerr << "object["<<ii<<"] - name = "<<scene.obj[ii]->name<<"\n";
  for (ii=0; ii<scene.mesh.size(); ii++)
    cerr << "mesh["<<ii<<"] - name = "<<scene.mesh[ii]->obj->name<<"\n";



  // build new ImageRM
  irm = new ImageRM;
  irm->pix.resize(ny,nx);
  irm->min=specMin;
  irm->max=specMax;
  irm->num=specNum;
  irm->spacing=(specMax-specMin)/(specNum-1.);


  // setup the materials
  int i;
  for (i=0; i<scene.obj.size(); i++) {
    RTMaterialHandle mtlh=scene.obj[i]->matl;
    if (mtlh->emitting) {	
      int j;
      for (j=0; j<irm->lightName.size(); j++) {
	if (irm->lightName[j]==mtlh->name) break;
      }
      if (j==irm->lightName.size()) {
	mtlh->lidx=j;
	irm->lightName.add(mtlh->name);
	irm->lightSpec.add(mtlh->emission);
      }
    }
    int j;
    for (j=0; j<irm->matlName.size(); j++) {
      if (irm->matlName[j]==mtlh->name) break;
    }
    if (j==irm->matlName.size()) {
      mtlh->midx=j;
      irm->matlName.add(mtlh->name);
      irm->matlSpec.add(mtlh->diffuse);
      irm->kd.add(1-mtlh->base->reflectivity);
      irm->ks.add(mtlh->base->reflectivity);
    }
  }

  cerr << "Here are my materials...\n";
  for (i=0; i<irm->matlName.size(); i++) {
    cerr << i <<" "<<irm->matlName[i]<<"\n";
  }

  cerr << "Here are my lights...\n";
  for (i=0; i<irm->lightName.size(); i++) {
    cerr << i <<" "<<irm->lightName[i]<<"\n";
  }


  int numSmp=ns;
  if (numSmp == 0) numSmp=1;
  int nl=irm->lightName.size();
  int nm=irm->matlName.size();
  for (int jj=0; jj<ny; jj++) {
    for (int ii=0; ii<nx; ii++) {
      irm->pix(jj,ii).nSamples=numSmp;
      DenseMatrix lm(nl,nm);
      DenseMatrix mm(nm,nm);
      lm.zero(); mm.zero();

      // diagonalize!!!
      int kk;
      for (kk=0; kk<nm; kk++) mm[kk][kk]=1;

      irm->pix(jj,ii).R=lm;
      irm->pix(jj,ii).S.resize(scene.numBounces);
      irm->pix(jj,ii).D.resize(scene.numBounces+1);
      for (kk=0; kk<scene.numBounces; kk++) {
	irm->pix(jj,ii).S[kk]=mm;
	irm->pix(jj,ii).D[kk]=lm;
      }
      irm->pix(jj,ii).D[scene.numBounces]=lm;
      irm->pix(jj,ii).S0.resize(nm);
      irm->pix(jj,ii).E.resize(nl);
      irm->pix(jj,ii).S0.initialize(0);
      irm->pix(jj,ii).E.initialize(0);
    }
  }
  irm->ka.set(300, 0);
  irm->ka.set(750, 0);
}    

void DRaytracer::rayTrace(int minx, int maxx, int miny, int maxy, 
			  double* image, unsigned char* rawImage,
			  double* xyz) {
  int x, y, b;
  for (y=miny;y<maxy;y++) {
    b=y*nx*3;
    for (x=minx;x<maxx;x++,b+=3) {
      Point p;
      Color c=singleTrace(topLeft+(camera.v*stepY*y)+(camera.u*stepX*x),
			  p, &(irm->pix(y,x)));
      image[b]=c.r();
      image[b+1]=c.g();
      image[b+2]=c.b();
      xyz[b]=p.x();
      xyz[b+1]=p.y();
      xyz[b+2]=p.z();
      //	    cerr << "color: "<<image[b]<<" "<<image[b+1]<<" "<<image[b+2]<<" ";
      rawImage[b]=Max(0.,Min(255.,image[b]*255));
      rawImage[b+1]=Max(0.,Min(255.,image[b+1]*255));
      rawImage[b+2]=Max(0.,Min(255.,image[b+2]*255));
    }
  }
}
    
void DRaytracer::rayTrace(double* image, unsigned char* rawImage, double* xyz){
  preRayTrace();
  rayTrace(0,nx,0,ny,image,rawImage,xyz);
}

inline void SWAP_INT(int &a, int &b) {int c=a; a=b; b=c;}

inline void SWAP_DOUBLE (double &a, double &b) {double c=a; a=b; b=c;}

void bldAndShuffle(Array1<int>& shuffle, MusilRNG& mr) {
  int smpl;
  for (smpl=0; smpl<shuffle.size(); smpl++) shuffle[smpl]=smpl;
  for (smpl=0; smpl<shuffle.size(); smpl++) {
    int rnd=mr()*shuffle.size();
    SWAP_INT(shuffle[rnd], shuffle[smpl]);
  }
}

void shufflePts(Array1<double>& arr, Array1<int>& loc) {
  for (int i=0; i<arr.size(); i++) {
    SWAP_DOUBLE(arr[i], arr[loc[i]]);
  }
}

void unitSquareToCircle(double R, Array1<double>& x, Array1<double>& y) {
  for (int smpl=0; smpl<x.size(); smpl++) {
    double theta=2*M_PI*x[smpl];
    double r=R*sqrt(y[smpl]);
    x[smpl]=r*cos(theta);
    y[smpl]=r*sin(theta);
  }
}

void DRaytracer::lensPixelSamples(Sample2D &s, Array1<double>& pixX, 
				  Array1<double>& pixY, Array1<double>& lensX, 
				  Array1<double>& lensY, Array1<double>& w,
				  Array1<int>& shuffle) {
    
  s.genSamples(pixX, pixY, w, pixX.size());

  //cerr << "apperture="<<camera.apperture<<"\n";
  if (camera.apperture != 0) {	// not pinhole
    s.genSamples(lensX, lensY, w, lensX.size());
    if (pixX.size()>1) {
      bldAndShuffle(shuffle, s.mr);
      shufflePts(lensX, shuffle);
      shufflePts(lensY, shuffle);
    }

    // Radius of the effective lens
    double R=(camera.fLength*camera.zoom)/camera.apperture;   
    //cerr << "R="<<R<<"\n";
    unitSquareToCircle(R, lensX, lensY);
  } else {		// pinhole
    for (int smpl=0; smpl<ns; smpl++) 
      lensX[smpl]=lensY[smpl]=0;
  }
}

#define DRaytracer_VERSION 2
void DRaytracer::io(Piostream& stream) {

  /*int version=*/stream.begin_class("DRaytracer", DRaytracer_VERSION);
  VoidStar::io(stream);
  SCIRun::Pio(stream, scene);
  SCIRun::Pio(stream, camera);
  if (0) {
    //    if (version < 2) {
    ns=0;
    nx=100;
    ny=100;
    specMin=300;
    specMax=800;
    specNum=11;
  } else {
    SCIRun::Pio(stream, ns);
    SCIRun::Pio(stream, nx);
    SCIRun::Pio(stream, ny);
    SCIRun::Pio(stream, specMin);
    SCIRun::Pio(stream, specMax);
    SCIRun::Pio(stream, specNum);
  }
  stream.end_class();
}

} // End namespace DaveW

