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

#ifndef SCI_DaveW_Datatypes_Sample2D_h
#define SCI_DaveW_Datatypes_Sample2D_h 1

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Math/MusilRNG.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::Array1;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;

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
	QuasiMonteCarlo
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

} // End namespace Datatypes
} // End namespace DaveW
//
// $Log$
// Revision 1.1  1999/08/23 02:52:58  dmw
// Dave's Datatypes
//
// Revision 1.2  1999/05/03 04:52:03  dmw
// Added and updated DaveW Datatypes/Modules
//
//
#endif
