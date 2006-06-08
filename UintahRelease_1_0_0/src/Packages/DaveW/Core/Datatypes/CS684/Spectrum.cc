
/*
 *  Spectrum.cc: Generate Spectrum points in a domain
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/CS684/Spectrum.h>
#include <Packages/DaveW/Core/Datatypes/CS684/xyz.h>
#include <iostream>
using std::cerr;

namespace DaveW {
using DaveW::Pio;
using SCIRun::Pio;
Spectrum::Spectrum()
{
}

Spectrum::Spectrum(const Spectrum &copy)
: wavelength(copy.wavelength), amplitude(copy.amplitude)
{
}

Spectrum::Spectrum(float *amps, double spacing, double min, int num) {
    wavelength.resize(num);
    amplitude.resize(num);
    double wl=min;
    for (int i=0; i<num; i++, wl+=spacing) {
	wavelength[i]=wl;
	amplitude[i]=amps[i];
    }
}

void buildXYZ(Array1<double>& xamp, double &xmin, double &xsp, 
              Array1<double>& yamp, double &ymin, double &ysp, 
              Array1<double>& zamp, double &zmin, double &zsp) {
    xsp=ysp=zsp=5;
    xmin=375;
    ymin=380;
    zmin=375;
    xamp.resize(82);
    yamp.resize(78);
    zamp.resize(52);
    int i;
    for (i=0; i<xamp.size(); i++)
        xamp[i]=xAmplitudes[i];
    for (i=0; i<yamp.size(); i++)
        yamp[i]=yAmplitudes[i];
    for (i=0; i<zamp.size(); i++)
        zamp[i]=zAmplitudes[i];
}

    
Spectrum::~Spectrum() {
}

void Spectrum::rediscretize(Array1<double> &newAmps, double newMin, 
			    double newMax) {
    newAmps.initialize(0);

    // make sure the Spectrum's wavelengths are evenly spaced...
    int even=1;	
    double oldSp=wavelength[1]-wavelength[0];
    for (int i=2; even && (i<wavelength.size()); i++) {
	if (wavelength[i]-wavelength[i-1] != oldSp) {
	    cerr << "unevenness at i="<<i<<": wavelength[i]="<<wavelength[i]<<" wavelength[i-1]="<<wavelength[i-1]<<"\n";
	    even=0;
	}
    }
    if (!even) {
	cerr << "Can't rediscretize non-evenly spaced spectra (yet).\n";
	return;
    }
    
    int oldNum=wavelength.size();
    int newNum=newAmps.size();
    double newSp=(newMax-newMin)/(newNum-1);
    double oldMin=wavelength[0];
    double oldMax=wavelength[wavelength.size()-1];

    double newIVal=newMin;
    for (int newIdx=0; newIdx<newNum; newIdx++, newIVal+=newSp) {
	double dd=(newIVal-oldMin)/oldSp;
	int idx=dd;
	double d=dd-idx;
	if (dd<=0) newAmps[newIdx]=amplitude[0];
	else if (dd>=(oldNum-1)) newAmps[newIdx]=amplitude[amplitude.size()-1];
	else {
	    newAmps[newIdx]=((1-d)*amplitude[idx]+d*amplitude[idx+1]);
	}
    }
}

double Spectrum::integrate(const Array1<double> &amps, double spacing, 
			   double min) {
    double max=min+(amps.size()-1)*spacing;
    double sum=0;
    for (int i=0; i<amplitude.size(); i++) {
	double a=amplitude[i];
	double wl=wavelength[i];
	if (wl<=min) sum+=amps[0]*a;
	else if (wl>=max) sum+=amps[amps.size()-1]*a;
	else {
	    double dd=(wl-min)/spacing;
	    int idx=dd;
	    double d=dd-idx;
	    sum+=((1-d)*amps[idx]+d*amps[idx+1])*a;
	}
    }
    return sum;   // sum*683
}

LiteSpectrum::LiteSpectrum()
{
min=0; max=0; num=0; spacing=0; vals=0;
}

LiteSpectrum::LiteSpectrum(double min, double max, int num, double spacing, 
			   double *vals) 
 : min(min), max(max), num(num), spacing(spacing), vals(vals)
{
}

LiteSpectrum::~LiteSpectrum() {
}

Point LiteSpectrum::xyz(double *xAmps, double *yAmps, double *zAmps) {
    double x=vectorDotProd(xAmps, vals, num)/num;
    double y=vectorDotProd(yAmps, vals, num)/num;
    double z=vectorDotProd(zAmps, vals, num)/num;
    return Point(x,y,z);
}

void vectorAddScale(Array1<double> &a, const Array1<double> &b, double s) {
    for (int i=0; i<a.size(); i++) a[i]+=b[i]*s;
}

void vectorScaleBy(Array1<double> &a, const Array1<double> &b) {
    for (int i=0; i<a.size(); i++) a[i]*=b[i];
}

void vectorScaleBy(Array1<double> &a, double s) {
    for (int i=0; i<a.size(); i++) a[i]*=s;
}

double vectorDotProd(const Array1<double> &a, const Array1<double> &b) {
    double sum=0;
    for (int i=0; i<a.size(); i++) sum+=a[i]*b[i];
    return sum;
}

double vectorDotProd(double *a, double *b, int num) {
    double sum=0;
    for (int i=0; i<num; i++) sum+=a[i]*b[i];
    return sum;
}

Color XYZ_to_RGB(const Point& p) {
    Color c;
    c.r(NTSC_rgb[0][0]*p.x()+NTSC_rgb[0][1]*p.y()+NTSC_rgb[0][2]*p.z());
    c.g(NTSC_rgb[1][0]*p.x()+NTSC_rgb[1][1]*p.y()+NTSC_rgb[1][2]*p.z());
    c.b(NTSC_rgb[2][0]*p.x()+NTSC_rgb[2][1]*p.y()+NTSC_rgb[2][2]*p.z());
    return c;
}
} // End namespace DaveW

namespace SCIRun {
using namespace DaveW;
void Pio(Piostream& stream, Spectrum& s)
{
  stream.begin_cheap_delim();
  Pio(stream, s.amplitude);
  Pio(stream, s.wavelength);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, LiteSpectrum& s)
{

  stream.begin_cheap_delim();
  Pio(stream, s.min);
  Pio(stream, s.max);
  Pio(stream, s.num);
  Pio(stream, s.spacing);
  if (stream.reading()) s.vals=new double[s.num];
  for (int i=0; i<s.num; i++)
    Pio(stream, s.vals[i]);
  stream.end_cheap_delim();
}

} //End namespace SCIRun


