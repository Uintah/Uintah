
/*
 *  ImageR.h: The ImageXYZ and ImageRM datatypes - used in the Raytracer
 *	     and Radioisity code.  These types are derived from the
 *	     VoidStar class.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_ImageR_h
#define SCI_DaveW_Datatypes_ImageR_h 1

#include <DaveW/Datatypes/CS684/Pixel.h>
#include <DaveW/Datatypes/CS684/Spectrum.h>
#include <SCICore/Datatypes/VoidStar.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::Array2;
using SCICore::Containers::clString;
using SCICore::Datatypes::DenseMatrix;
using SCICore::Datatypes::VoidStar;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class ImageXYZ : public VoidStar {
public:
    Array2<double> xyz;
public:
    ImageXYZ();
    ImageXYZ(const ImageXYZ& copy);
    ImageXYZ(const Array2<double>& xyz);
    virtual ~ImageXYZ();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
    
class ImageRM : public VoidStar {
public:
    Array2<Pixel> pix;
    Array1<Spectrum> lightSpec;
    DenseMatrix *LS;
    Array1<clString> lightName;
    Array1<Spectrum> matlSpec;
    DenseMatrix *MS;
    Array1<clString> matlName;
    Array1<double> kd;
    Array1<double> ks;
    Spectrum ka;
    DenseMatrix *KAS;
    int min, max, num;
    double spacing;
public:
    ImageRM();
    ImageRM(const ImageRM& copy);
    ImageRM(const Array2<Pixel>& p, const Array1<Spectrum>& ls, 
	    const Array1<clString>& ln, const Array1<Spectrum>& ms,
	    const Array1<clString>& mn, const Array1<double>& d,
	    const Array1<double>& s, const Spectrum& a);
    void getSpectraMinMax(double &min, double &max);
    void bldSpecLightMatrix(int lindex);
    void bldSpecMaterialMatrix(int mindex);
    void bldSpecAmbientMatrix();
    void bldSpecMatrices();
    void bldPixelR();
    void bldPixelSpectrum();
    void bldPixelXYZandRGB();
    virtual ~ImageRM();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace DaveW
//
// $Log$
// Revision 1.2  1999/08/25 03:35:47  sparker
// *** empty log message ***
//
// Revision 1.1  1999/08/23 02:52:56  dmw
// Dave's Datatypes
//
// Revision 1.2  1999/05/03 04:52:00  dmw
// Added and updated DaveW Datatypes/Modules
//
//
#endif
