/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/GeomColormapInterface.h>

/* changed to work less stupid with transfer functions
 * Peter-Pike Sloan
 */

namespace SCIRun {


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

} // End namespace SCIRun


#endif
