
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

#include <Datatypes/ColorMap.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* make_ColorMap()
{
    return scinew ColorMap;
}

PersistentTypeID ColorMap::type_id("ColorMap", "Datatype", make_ColorMap);

ColorMap::ColorMap()
: min(-1), max(1), colors(50), non_diffuse_constant(0),type(0),
  rawRed(0),rawGreen(0),rawBlue(0),rawAlpha(0)
{
   build_default();
}

ColorMap::ColorMap(int nlevels, double min, double max, int shortrange )
: min(min), max(max), colors(nlevels), non_diffuse_constant(0),type(0),
  rawRed(0),rawGreen(0),rawBlue(0),rawAlpha(0)
{
}

ColorMap::ColorMap(const ColorMap& copy)
: min(copy.min), max(copy.max), colors(copy.colors),
  non_diffuse_constant(copy.non_diffuse_constant),
  rawRed(0),rawGreen(0),rawBlue(0),rawAlpha(0)
{
}

inline Color FindColor(const Array1<Color>& c,const Array1<float>& s,float t)
{
  int j=0;

  if (t<=s[0])
    return c[0];
  if (t>= s[s.size()-1])
    return c[1];

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
    return c[1];

  // t is within the interval...

  while((j < c.size()) && (t > s[j])) {
    j++;
  }

  float slop = (s[j] - t)/(s[j]-s[j-1]);

  return c[j-1]*slop + c[j]*(1.0-slop);
}

ColorMap::ColorMap(const Array1<Color>& rgb, Array1<float>& rgbT,
	     const Array1<float>& ialpha, const Array1<float>& alphaT,
		   const int size):min(0),max(1),non_diffuse_constant(1),
		   type(1),rcolors(size),alphas(size),colors(size),
		   rawRed(0),rawGreen(0),rawBlue(0),rawAlpha(0)
{
  SetRaw(rgb,rgbT,ialpha,alphaT,size);
  
  Color ambient(0,0,0),specular(0.7,0.7,0.7);
  for(int i=0;i<size;i++) {
    colors[i] = scinew Material(ambient,rcolors[i],specular, 10);
  }
}

void ColorMap::SetRaw(const Array1<Color>& rgb, Array1<float>& rgbT,
		      const Array1<float>& ialpha, 
		      const Array1<float>& alphaT,
		      const int size)
{
  float mul = 1.0/(size-1);

  if (!rawRed) {
    rawRed = new double[size];
    rawGreen = new double[size];
    rawBlue = new double[size];
    rawAlpha = new double[size];

  }
  for(int i=0;i<size;i++) {
    rcolors[i] = FindColor(rgb,rgbT,i*mul);
  }
  
  for(i=0;i<size;i++) {
    alphas[i] = FindAlpha(ialpha,alphaT,i*mul);
    rawRed[i] = rcolors[i].r();
    rawGreen[i] = rcolors[i].g();
    rawBlue[i] = rcolors[i].b();
    rawAlpha[i] = alphas[i];
  }

  // this time i want the ramp information
  rawRampAlpha = new Array1<float>;
  *rawRampAlpha = ialpha;

  rawRampAlphaT = new Array1<float>;
  *rawRampAlphaT = alphaT;
  
  // convert float scalar values to ints
  rawRampColorT = new Array1<float>;
  *rawRampColorT = rgbT;
  
  rawRampColor = new Array1<Color>;
  *rawRampColor = rgb;
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

#define COLORMAP_VERSION 1

void ColorMap::io(Piostream& stream)
{
    /*int version=*/ stream.begin_class("ColorMap", COLORMAP_VERSION);
    Pio(stream, colors);
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


#ifdef __GNUG__

#include <Classlib/LockingHandle.cc>
template class LockingHandle<ColorMap>;

#include <Classlib/Array1.cc>
template void Pio(Piostream&, Array1<MaterialHandle>&);
template void Pio(Piostream&, MaterialHandle&);

#endif

#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy_(Piostream& p1, Array1<MaterialHandle>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

