
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

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Geom/Material.h>
#include <Geom/Color.h>

class ColorMap;
typedef LockingHandle<ColorMap> ColorMapHandle;

/* changed to work less stupid with transfer functions
 * Peter-Pike Sloan
 */

class ColorMap : public Datatype {
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

    unsigned int            flag;
    
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

    MaterialHandle& lookup(double value);
    MaterialHandle& lookup2(double value);

    virtual ~ColorMap();
    virtual ColorMap* clone();

    void build_default();
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
