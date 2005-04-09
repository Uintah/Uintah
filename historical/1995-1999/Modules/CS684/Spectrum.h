/*
 *  Spectrum.h: Generate sample points in a domain
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_Spectrum_h
#define SCI_project_Spectrum_h 1

#include <Classlib/Array1.h>
#include <Classlib/Persistent.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>

#include <Modules/CS684/xyz.h>

class Spectrum {
public:
    Array1<double> wavelength;
    Array1<double> amplitude;
public:
    Spectrum();
    Spectrum(const Spectrum &copy);
    Spectrum(float *amps, double spacing, double min, int num);
    ~Spectrum();
    void rediscretize(Array1<double> &newAmps, double newMin, double newMax);
    double integrate(const Array1<double> &amps, double spacing, double min);
    inline void set(double w, double a) {wavelength.add(w); amplitude.add(a);}
    inline void clear() {wavelength.resize(0); amplitude.resize(0);}
};

class LiteSpectrum {
public:
    double min;
    double max;
    int num;
    double spacing;
    double *vals;
public:
    LiteSpectrum();
    LiteSpectrum(double min, double max, int num, double spacing, 
		 double *vals);
    ~LiteSpectrum();
    Point xyz(double *, double *, double *);
};

void Pio(Piostream&, Spectrum&);
void Pio(Piostream&, LiteSpectrum&);

Color XYZ_to_RGB(const Point &p);

void vectorAddScale(Array1<double> &a, const Array1<double> &b, double s);
void vectorScaleBy(Array1<double> &a, const Array1<double> &b);
void vectorScaleBy(Array1<double> &a, double s);
double vectorDotProd(const Array1<double> &a, const Array1<double> &b);
double vectorDotProd(double *a, double *b, int num);

#endif
