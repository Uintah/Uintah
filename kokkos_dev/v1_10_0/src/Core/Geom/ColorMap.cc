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

#include <Core/Geom/ColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/PersistentSTL.h>
#include <iostream>
#include <algorithm>

using std::cerr;
using std::endl;

namespace SCIRun {


static Persistent* make_ColorMap()
{
  return scinew ColorMap;
}


PersistentTypeID ColorMap::type_id("ColorMap", "Datatype", make_ColorMap);


ColorMap::ColorMap()
  : raw1d(0),
    min_(-1),
    max_(1),
    colors_(50),
    pre_mult_alpha_p_(false),
    scaled_p_(false)
{
  build_default();
}


ColorMap::ColorMap(int nlevels, double min, double max, int /*shortrange */)
  : raw1d(0),
    min_(min),
    max_(max),
    colors_(nlevels),
    pre_mult_alpha_p_(false),
    scaled_p_(false)
{
}


ColorMap::ColorMap(const ColorMap& copy)
  : rawRampAlpha(copy.rawRampAlpha),
    rawRampAlphaT(copy.rawRampAlphaT),
    rawRampColor(copy.rawRampColor),
    rawRampColorT(copy.rawRampColorT),
    raw1d(copy.raw1d),
    min_(copy.min_),
    max_(copy.max_),
    colors_(copy.colors_),
    rcolors_(copy.rcolors_),
    alphas_(copy.alphas_),
    pre_mult_alpha_p_(copy.pre_mult_alpha_p_),
    scaled_p_(copy.scaled_p_)
{
  Build1d();
}


ColorMap::ColorMap(const vector<Color>& rgb,
		   const vector<float>& rgbT,
		   const vector<float>& ialpha,
		   const vector<float>& alphaT,
		   const int size)
  : raw1d(0),
    min_(-1),
    max_(1),
    colors_(size),
    rcolors_(size),
    pre_mult_alpha_p_(false),
    scaled_p_(false)
{
  SetRaw(rgb,rgbT,ialpha,alphaT,size);
  
  Color ambient(0,0,0),specular(0.7,0.7,0.7);
  //  cerr << "Constructor for ColorMap: " << size << endl;
  for(int i=0;i<size;i++)
  {
    colors_[i] = scinew Material(ambient,rcolors_[i],specular, 10);
    colors_[i]->transparency = alphas_[i];
  }
}


Color
ColorMap::FindColor(double t)
{
  vector<float>::iterator location = 
    std::lower_bound(rawRampColorT.begin(), rawRampColorT.end(), t);
  if (location == rawRampColorT.begin())
  {
    return rawRampColor[0];
  }
  else if (location == rawRampColorT.end())
  {
    return rawRampColor[rawRampColor.size()-1];
  }
  else
  {
    unsigned int j = location - rawRampColorT.begin();
    const double slop =
      (rawRampColorT[j] - t)/(rawRampColorT[j]-rawRampColorT[j-1]);

    return rawRampColor[j-1]*slop + rawRampColor[j]*(1.0-slop);
  }
}


double
ColorMap::FindAlpha(double t)
{
  vector<float>::iterator location = 
    std::lower_bound(rawRampAlphaT.begin(), rawRampAlphaT.end(), t);
  if (location == rawRampAlphaT.begin())
  {
    return rawRampAlpha[0];
  }
  else if (location == rawRampAlphaT.end())
  {
    return rawRampAlpha[rawRampAlpha.size()-1];
  }
  else
  {
    unsigned int j = location - rawRampAlphaT.begin();
    const double slop =
      (rawRampAlphaT[j] - t)/(rawRampAlphaT[j]-rawRampAlphaT[j-1]);

    return rawRampAlpha[j-1]*slop + rawRampAlpha[j]*(1.0-slop);
  }
}


void
ColorMap::Build1d(const int size)
{
  int i;
  float mul = 1.0/(size-1);

  rcolors_.resize(size);
  alphas_.resize(size);
  for(i=0;i<size;i++)
  {
    rcolors_[i] = FindColor(i*mul);
    alphas_[i] = FindAlpha(i*mul);
  }
  
  const int TEX1D_SIZE=256;
  if (!raw1d)
  {
    raw1d = scinew unsigned char[4*TEX1D_SIZE];
  }

  mul = 1.0/(TEX1D_SIZE-1);

  if (pre_mult_alpha_p_)
  {

    for (i=0;i<TEX1D_SIZE;i++)
    {
      const Color c = FindColor(i*mul);
      const double al = FindAlpha(i*mul);
      raw1d[i*4 + 0] = (unsigned char)(c.r()*al*255);
      raw1d[i*4 + 1] = (unsigned char)(c.g()*al*255);
      raw1d[i*4 + 2] = (unsigned char)(c.b()*al*255);
      raw1d[i*4 + 3] = (unsigned char)(al*255);
    }  
  }
  else
  { // don't pre-multiply the alpha value...
    for(i=0;i<TEX1D_SIZE;i++)
    {
      const Color c = FindColor(i*mul);
      const double al = FindAlpha(i*mul);
      raw1d[i*4 + 0] = (unsigned char)(c.r()*255);
      raw1d[i*4 + 1] = (unsigned char)(c.g()*255);
      raw1d[i*4 + 2] = (unsigned char)(c.b()*255);
      raw1d[i*4 + 3] = (unsigned char)(al*255);
    }  
  }
}



void
ColorMap::SetRaw(const vector<Color>& rgb,
		 const vector<float>& rgbT,
		 const vector<float>& ialpha,
		 const vector<float>& alphaT,
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


void
ColorMap::build_default()
{
  double hue_min=0;
  double hue_max=300;
  double hue_range=hue_max-hue_min;
  double sat=1;
  double val=1;
  const unsigned int nl=colors_.size();
  for(unsigned int i=0;i<nl;i++)
  {
    const double hue = double(i)/double(nl-1)*hue_range+hue_min;
    Color base(HSVColor(hue, sat, val));
    Color ambient(base*.1);
    Color diffuse(base);
    Color specular(1,1,1);
    colors_[i]=scinew Material(ambient, diffuse, specular, 10);
  }
}    


#define COLORMAP_VERSION 4


void
ColorMap::io(Piostream& stream)
{

  int version= stream.begin_class("ColorMap", COLORMAP_VERSION);
  Pio(stream, colors_);
  if ( version > 2 )
    Pio(stream, units);
  if ( version > 1 )
  {
    Pio(stream,rawRampAlpha);
    Pio(stream,rawRampAlphaT);
    Pio(stream,rawRampColor);
    Pio(stream,rawRampColorT);
      
    if ( stream.reading() )
    {
      Build1d();
    }
  }
  stream.end_class();
}


MaterialHandle&
ColorMap::lookup(double value)
{
  int idx=int((colors_.size()-1)*(value-min_)/(max_-min_));
  if(idx<0)
    idx=0;
  else if(idx > (int)colors_.size()-1)
    idx=colors_.size()-1;
  return colors_[idx];
}


MaterialHandle&
ColorMap::lookup2(double nvalue)
{
  int idx;
  idx=int((colors_.size()-1)*nvalue);

  if(idx<0)
    idx=0;
  else if(idx > (int)colors_.size()-1)
    idx=colors_.size()-1;
  return colors_[idx];
}


double
ColorMap::getMin() const
{
  return min_;
}


double
ColorMap::getMax() const
{
  return max_;
}


} // End namespace SCIRun


