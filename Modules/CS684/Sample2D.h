/*
 *  Sample2D.h: Generate sample points in a domain
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_Sample2D_h
#define SCI_project_Sample2D_h 1

#include <Classlib/Array1.h>
#include <Classlib/String.h>
#include <Math/MusilRNG.h>

class Sample2D {
    void genSamplesJittered(Array1<double>&x, Array1<double>&y, 
			    Array1<double>&w);
    void genSamplesRandom(Array1<double>&x, Array1<double>&y,
			    Array1<double>&w);
    void genSamplesRegular(Array1<double>&x, Array1<double>&y,
			    Array1<double>&w);
    void genSamplesPoisson(Array1<double>&x, Array1<double>&y,
			    Array1<double>&w);
    void genSamplesMultiJittered(Array1<double>&x, Array1<double>&y,
			    Array1<double>&w);
public:
    MusilRNG mr;
    enum Representation {
	Jittered,
	Random,
	Regular,
	Poisson,
	MultiJittered,
	QuasiMonteCarlo,
    };
    void genSamplesQuasiMonteCarlo(Array1<double>&x, Array1<double>&y,
				   int start, int stop, int max);
private:
    Representation rep;
public:
    Sample2D(int mr=0);
    Sample2D(const Sample2D &copy);
    ~Sample2D();

    clString getMethod() const;
    void setMethod(const clString& m);
    void genSamples(Array1<double>& x, Array1<double>& y, Array1<double>& w,
		    int ns);
};

#endif
