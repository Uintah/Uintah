
/*
 *  ColorMap.h: ColorMap definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_ColorMap_h
#define SCI_project_ColorMap_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomColormapInterface.h>

/* changed to work less stupid with transfer functions
 * Peter-Pike Sloan
 */

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::GeomSpace::MaterialHandle;
using SCICore::GeomSpace::Color;
using SCICore::GeomSpace::GeomColormapInterface;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

class ColorMap;
typedef LockingHandle<ColorMap> ColorMapHandle;

class SCICORESHARE ColorMap : public Datatype, public GeomColormapInterface {
public:
    int type; // 0 is stupid, 1 is not stupid
    double min;
    double max;
    Array1<MaterialHandle> colors;

    Array1<Color> rcolors;// make this big...
    Array1<float> alphas; // alphas - same size

//    unsigned char*        raw;  // raw data... 

    double      *rawRed;
    double      *rawGreen;
    double      *rawBlue;
    double      *rawAlpha;

    Array1<float> rawRampAlpha;
    Array1<float> rawRampAlphaT;
    Array1<Color> rawRampColor;
    Array1<float> rawRampColorT;
  
    unsigned int            flag;

    unsigned int            pre_mult_alpha; // set if you want it...
    
    unsigned char*          raw1d;

    int non_diffuse_constant;   // 1 if non diffuse materials are constant
    ColorMap();
    ColorMap(const ColorMap&);
    ColorMap(int nlevels, double min, double max, int shortrange=0);

    ColorMap(const Array1<Color>& rgb, Array1<float>& rgbT,
	     const Array1<float>& alphas, const Array1<float>& alphaT,
	     const int size=2000);

    void SetRaw(const Array1<Color>& rgb, Array1<float>& rgbT,
		const Array1<float>& alphas, const Array1<float>& alphaT,
		const int size=2000);

    void Build1d(const int size=256);


  bool IsScaled(){ return scaled;} // are the colors scaled to some data?
  void Scale(double min, double max){
    this->min = min; this->max = max; scaled = true;}
  void ResetScale() { min = -1; max = 1; scaled = false; }

    MaterialHandle& lookup(double value);
    MaterialHandle& lookup2(double value);

    virtual ~ColorMap();
    virtual ColorMap* clone();

    void build_default();
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    virtual double getMin();
    virtual double getMax();
private:
  bool scaled;
   
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:31  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:19  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:46  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:36  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:04  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

#endif
