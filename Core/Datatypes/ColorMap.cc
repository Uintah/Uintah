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

#include <Core/Datatypes/ColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace SCIRun {


static Persistent* make_ColorMap()
{
    return scinew ColorMap;
}

PersistentTypeID ColorMap::type_id("ColorMap", "Datatype", make_ColorMap);

ColorMap::ColorMap()
: type(0),min(-1), max(1), colors(50), rawRed(0),rawGreen(0),rawBlue(0),
  rawAlpha(0),pre_mult_alpha(0),raw1d(0),non_diffuse_constant(0),scaled(false)
{
   build_default();
}

ColorMap::ColorMap(int nlevels, double min, double max, int /*shortrange */)
: type(0),min(min), max(max), colors(nlevels), rawRed(0),rawGreen(0),
  rawBlue(0),rawAlpha(0),pre_mult_alpha(0),raw1d(0),non_diffuse_constant(0),
  scaled(false)
{
}

ColorMap::ColorMap(const ColorMap& copy)
: type(copy.type), min(copy.min), max(copy.max), colors(copy.colors),
  rcolors(copy.rcolors), alphas(copy.alphas), rawRed(0), rawGreen(0),
  rawBlue(0), rawAlpha(0), rawRampAlpha(copy.rawRampAlpha), 
  rawRampAlphaT(copy.rawRampAlphaT), rawRampColor(copy.rawRampColor),
  rawRampColorT(copy.rawRampColorT), flag(copy.flag),
  pre_mult_alpha(copy.pre_mult_alpha), raw1d(copy.raw1d),
  non_diffuse_constant(copy.non_diffuse_constant), scaled(copy.scaled)
{
  Build1d();
}

ColorMap::ColorMap(const Array1<Color>& rgb, Array1<float>& rgbT,
	     const Array1<float>& ialpha, const Array1<float>& alphaT,
		   const int size)
: type(1),min(-1), max(1), colors(size), rcolors(size), rawRed(0),rawGreen(0),
  rawBlue(0),rawAlpha(0),pre_mult_alpha(0),raw1d(0),non_diffuse_constant(1),
  scaled(false)
{
  SetRaw(rgb,rgbT,ialpha,alphaT,size);
  
  Color ambient(0,0,0),specular(0.7,0.7,0.7);
  //  cerr << "Constructor for ColorMap: " << size << endl;
  for(int i=0;i<size;i++) {
    colors[i] = scinew Material(ambient,rcolors[i],specular, 10);
    colors[i]->transparency = alphas[i];
  }

  
}

inline Color FindColor(const Array1<Color>& c,const Array1<float>& s,float t)
{
  int j=0;

  if (t<=s[0])
    return c[0];
  if (t>= s[s.size()-1])
    return c[c.size()-1];

  // t is within the interval...

  while((j < c.size()) && (t > s[j])) {
    j++;
  }

  double slop = (s[j] - t)/(s[j]-s[j-1]);

  return c[j-1]*slop + c[j]*(1.0-slop);
  
}

inline float FindAlpha(const Array1<float>& c,const Array1<float>& s,float t)
{
  int j=0;

  if (t<=s[0])
    return c[0];
  if (t>= s[s.size()-1])
    return c[c.size()-1];

  // t is within the interval...

  while((j < c.size()) && (t > s[j])) {
    j++;
  }

  float slop = (s[j] - t)/(s[j]-s[j-1]);

  return c[j-1]*slop + c[j]*(1.0-slop);
}

void ColorMap::Build1d(const int size)
{
  int i;
  float mul = 1.0/(size-1);

  rcolors.resize(size);
  alphas.resize(size);
  for(i=0;i<size;i++) {
    rcolors[i] = FindColor(rawRampColor,rawRampColorT,i*mul);
    alphas[i] = FindAlpha(rawRampAlpha,rawRampAlphaT,i*mul);
  }
  
#if 0
  if (!rawRed) {
    rawRed = new double[size];
    rawGreen = new double[size];
    rawBlue = new double[size];
    rawAlpha = new double[size];

  }
  
  for(i=0;i<size;i++) {
    alphas[i] = FindAlpha(rawRampAlpha,rawRampAlphaT,i*mul);
    rawRed[i] = rcolors[i].r();
    rawGreen[i] = rcolors[i].g();
    rawBlue[i] = rcolors[i].b();
    rawAlpha[i] = alphas[i];
  }
#endif
  const int TEX1D_SIZE=256;
  if (!raw1d) {
    raw1d = scinew unsigned char[4*TEX1D_SIZE];
  }

  mul = 1.0/(TEX1D_SIZE-1);

  if (pre_mult_alpha) {

    for(i=0;i<TEX1D_SIZE;i++) {
      Color c = FindColor(rawRampColor,rawRampColorT,i*mul);
      double al = FindAlpha(rawRampAlpha,rawRampAlphaT,i*mul);
      raw1d[i*4 + 0] = (unsigned char)(c.r()*al*255);
      raw1d[i*4 + 1] = (unsigned char)(c.g()*al*255);
      raw1d[i*4 + 2] = (unsigned char)(c.b()*al*255);
      raw1d[i*4 + 3] = (unsigned char)(al*255);
    }  
  } else { // don't pre-multiply the alpha value...
    for(i=0;i<TEX1D_SIZE;i++) {
      Color c = FindColor(rawRampColor,rawRampColorT,i*mul);
      double al = FindAlpha(rawRampAlpha,rawRampAlphaT,i*mul);
      raw1d[i*4 + 0] = (unsigned char)(c.r()*255);
      raw1d[i*4 + 1] = (unsigned char)(c.g()*255);
      raw1d[i*4 + 2] = (unsigned char)(c.b()*255);
      raw1d[i*4 + 3] = (unsigned char)(al*255);
    }  
  }
}

void ColorMap::SetRaw(const Array1<Color>& rgb, Array1<float>& rgbT,
		      const Array1<float>& ialpha, 
		      const Array1<float>& alphaT,
		      const int size)
{
  // this time i want the ramp information
  rawRampAlpha = ialpha;

  rawRampAlphaT = alphaT;
  
  // convert float scalar values to ints
  rawRampColorT = rgbT;
  
  rawRampColor = rgb;

  Build1d(size);
}

ColorMap::~ColorMap()
{
}

ColorMap* ColorMap::clone()
{
    return scinew ColorMap(*this);
}

void ColorMap::build_default() {
	double hue_min=0;
	double hue_max=300;
	double hue_range=hue_max-hue_min;
	double sat=1;
	double val=1;
	int nl=colors.size();
	for(int i=0;i<nl;i++){
	    double hue=double(i)/double(nl-1)*hue_range+hue_min;
	    Color base(HSVColor(hue, sat, val));
	    Color ambient(base*.1);
	    Color diffuse(base);
	    Color specular(1,1,1);
	    colors[i]=scinew Material(ambient, diffuse, specular, 10);
	}
}    

#define COLORMAP_VERSION 2

void ColorMap::io(Piostream& stream)
{

    int version= stream.begin_class("ColorMap", COLORMAP_VERSION);
    Pio(stream, colors);
    if ( version > 1 ) {
      Pio(stream,rawRampAlpha);
      Pio(stream,rawRampAlphaT);
      Pio(stream,rawRampColor);
      Pio(stream,rawRampColorT);
      
      if ( stream.reading() ) {
	Build1d();
      }
    }
    stream.end_class();
}

MaterialHandle& ColorMap::lookup(double value)
{
    int idx=int((colors.size()-1)*(value-min)/(max-min));
    if(idx<0)
	idx=0;
    else if(idx > colors.size()-1)
	idx=colors.size()-1;
    return colors[idx];
}

MaterialHandle& ColorMap::lookup2(double nvalue)
{
    int idx;
    idx=int((colors.size()-1)*nvalue);

    if(idx<0)
        idx=0;
    else if(idx > colors.size()-1)
        idx=colors.size()-1;
    return colors[idx];
}

double ColorMap::getMin()
{
    return min;
}

double ColorMap::getMax()
{
    return max;
}

} // End namespace SCIRun


