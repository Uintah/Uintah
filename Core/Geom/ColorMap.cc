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
  : rawRampAlpha_(),
    rawRampAlphaT_(),
    rawRampColor_(),
    rawRampColorT_(),
    rawRGBA_(0),
    units(),
    min_(-1),
    max_(1),
    colors_(),
    scaled_p_(false),
    blend_p_(false)
{
  ColorMap(256,-1.0, 1.0);
}


ColorMap::ColorMap(int nlevels, double min, double max)
  : rawRampAlpha_(),
    rawRampAlphaT_(),
    rawRampColor_(),
    rawRampColorT_(),
    rawRGBA_(0),
    units(),
    min_(min),
    max_(max),
    colors_(nlevels),
    scaled_p_(false),
    blend_p_(false)
{
  // Build Default Colormap
  const unsigned int num=50;
  const double hue_min=0, hue_max=300;
  const double hue_dt = (hue_max - hue_min)/(num-1);

  for(unsigned int i=0;i<num;i++)
  {
    const double hue = i*hue_dt+hue_min;
    rawRampColor_.push_back(HSVColor(hue, 1,1));
    rawRampColorT_.push_back(i /(num-1.0));
  }
  rawRampAlphaT_.push_back(0.0);
  rawRampAlphaT_.push_back(1.0);
  rawRampAlpha_.push_back(1.0);
  rawRampAlpha_.push_back(1.0);
  Build1d(256);
}


ColorMap::ColorMap(const ColorMap& copy)
  : rawRampAlpha_(copy.rawRampAlpha_),
    rawRampAlphaT_(copy.rawRampAlphaT_),
    rawRampColor_(copy.rawRampColor_),
    rawRampColorT_(copy.rawRampColorT_),
    rawRGBA_(0),
    min_(copy.min_),
    max_(copy.max_),
    colors_(copy.colors_),
    scaled_p_(copy.scaled_p_),
    blend_p_(copy.blend_p_)
{
  Build1d(256);
}


ColorMap::ColorMap(const vector<Color>& rgb,
		   const vector<float>& rgbT,
		   const vector<float>& ialpha,
		   const vector<float>& alphaT) 
 : rawRampAlpha_(ialpha),
   rawRampAlphaT_(alphaT),
   rawRampColor_(rgb),
   rawRampColorT_(rgbT),
   rawRGBA_(0),
   min_(-1),
   max_(1),
   colors_(rgb.size()),
   scaled_p_(false),
   blend_p_(false)
{
  Build1d(256);
}

void
ColorMap::set_blend(bool blend)
{
  if (blend_p_ == blend) return;
  blend_p_ = blend;
  Build1d(256);
}
  

static unsigned char COLOR_FTOB(double c)
{
  int tmp = (int)(c * 255.0 + 0.5);
  if (tmp > 255) tmp = 255;
  else if (tmp < 0) tmp = 0;
  return (unsigned char)tmp;
}


void
ColorMap::Build1d(const int size)
{
  double t = 0;
  double dt = 1.0 / (size-1);

  int cIdx = 0;
  double CT0 = rawRampColorT_[cIdx];
  double CT1 = rawRampColorT_[cIdx+1];
  Color C0 = rawRampColor_[cIdx];
  Color C1 = rawRampColor_[cIdx+1];
  const int cSize = rawRampColor_.size()-1;

  
  int aIdx = 0;
  double AT0 = rawRampAlphaT_[aIdx];
  double AT1 = rawRampAlphaT_[aIdx+1];
  double A0 = rawRampAlpha_[aIdx];
  double A1 = rawRampAlpha_[aIdx+1];
  const int aSize = rawRampAlpha_.size()-1;
  Color ambient(0,0,0),specular(0.7,0.7,0.7);

  if (rawRGBA_) { delete rawRGBA_; rawRGBA_ = 0; }
  if (!rawRGBA_) rawRGBA_ = scinew unsigned char[4*size];
  
  colors_.clear();
  colors_.resize(size);

  // Blend if 256 colors.
  set_blend(rawRampColor_.size() == 256);

  for (int i = 0; i < size; i++) {
    colors_[i] = scinew Material();
    
    MaterialHandle &color = colors_[i];
    
    if (!blend_p_) {
      color->diffuse = C0;      
      color->transparency = A0;
    } else {
      color->diffuse = C1 * (t - CT0) / (CT1 - CT0) + 
	C0 * (CT1 - t) / (CT1 - CT0);
    //cerr << "cIdx " << cIdx << "  rCsz " << rawRampColor_.size() << " rCTsz " << rawRampColorT_.size() << "  CT0 " << CT0 << "  CT1 " << CT1 << "  R: " << color->diffuse.r() << "  G: " << color->diffuse.g() << "  B: " << color->diffuse.g() << std::endl;

      color->transparency = A1 * (t - AT0) / (AT1 - AT0) + 
	A0 * (AT1 - t) / (AT1 - CT0);
    }

    color->ambient = ambient;
    color->specular = specular;
    
    rawRGBA_[i*4+0] = COLOR_FTOB(color->diffuse.r());
    rawRGBA_[i*4+1] = COLOR_FTOB(color->diffuse.g());
    rawRGBA_[i*4+2] = COLOR_FTOB(color->diffuse.b());
    rawRGBA_[i*4+3] = COLOR_FTOB(color->transparency);

    t += dt;
    if (cIdx < cSize && t >= CT1) {
      cIdx++;
      CT0 = rawRampColorT_[cIdx];
      CT1 = rawRampColorT_[cIdx+1];
      C0 = rawRampColor_[cIdx];
      C1 = rawRampColor_[cIdx+1];
    }
    
    if (aIdx < aSize && t >= AT1) { 
      aIdx++;
      AT0 = rawRampAlphaT_[aIdx];
      AT1 = rawRampAlphaT_[aIdx+1];
      A0 = rawRampAlpha_[aIdx];
      A1 = rawRampAlpha_[aIdx+1];
    }
  }
}



void
ColorMap::SetRaw(const vector<Color>& rgb,
		 const vector<float>& rgbT,
		 const vector<float>& ialpha,
		 const vector<float>& alphaT)
{
  rawRampColor_ = rgb;
  rawRampColorT_= rgbT;
  rawRampAlpha_ = ialpha;
  rawRampAlphaT_= alphaT;
  Build1d(256);
}


ColorMap::~ColorMap()
{
}


ColorMap* ColorMap::clone()
{
  return scinew ColorMap(*this);
}


Color
ColorMap::getColor(double t)
{
  return colors_[int(t*(colors_.size()-1))]->diffuse;
}

double
ColorMap::getAlpha(double t)
{
  return colors_[int(t*(colors_.size()-1))]->transparency;
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
    Pio(stream,rawRampAlpha_);
    Pio(stream,rawRampAlphaT_);
    Pio(stream,rawRampColor_);
    Pio(stream,rawRampColorT_);
      
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


