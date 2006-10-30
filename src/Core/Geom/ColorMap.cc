/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <algorithm>

using std::cerr;
using std::endl;

namespace SCIRun {


Persistent* ColorMap::maker()
{
  return scinew ColorMap;
}


PersistentTypeID ColorMap::type_id("ColorMap", "Datatype", maker);


GeomColormapInterface::~GeomColormapInterface()
{
}


// Used by maker function above
ColorMap::ColorMap()
  : rawRampAlpha_(),
    rawRampAlphaT_(),
    rawRampColor_(),
    rawRampColorT_(),
    resolution_(0),
    units_(),
    min_(-1.0),
    max_(1.0),
    is_scaled_(false),
    materials_()
{
}


ColorMap::ColorMap(const ColorMap& copy)
  : rawRampAlpha_(copy.rawRampAlpha_),
    rawRampAlphaT_(copy.rawRampAlphaT_),
    rawRampColor_(copy.rawRampColor_),
    rawRampColorT_(copy.rawRampColorT_),
    resolution_(copy.resolution_),
    min_(copy.min_),
    max_(copy.max_),
    is_scaled_(copy.is_scaled_),
    materials_()
{
  for (unsigned int i = 0; i < 256*4; ++i) {
    rawRGBA_[i] = copy.rawRGBA_[i];
  }
  build_materials_from_rgba();
}


ColorMap::ColorMap(const vector<Color>& rgb,
		   const vector<float>& rgbT,
		   const vector<float>& ialpha,
		   const vector<float>& alphaT,
		   unsigned int res) 
 : rawRampAlpha_(ialpha),
   rawRampAlphaT_(alphaT),
   rawRampColor_(rgb),
   rawRampColorT_(rgbT),
   resolution_(res),
   min_(-1.0),
   max_(1.0),
   is_scaled_(false),
   materials_()
{
  build_rgba_from_ramp();
  build_materials_from_rgba();
}


ColorMap::ColorMap(const float *rgba)
 : rawRampAlpha_(0),
   rawRampAlphaT_(0),
   rawRampColor_(0),
   rawRampColorT_(0),
   resolution_(0),
   min_(-1.0),
   max_(1.0),
   is_scaled_(false),
   materials_()
{
  for (unsigned int i = 0; i < 256*4; ++i) {
    rawRGBA_[i] = rgba[i];
  }
 
  build_ramp_from_rgba();
  build_materials_from_rgba();
}


ColorMap::~ColorMap()
{
}


ColorMap* ColorMap::clone()
{
  return scinew ColorMap(*this);
}




// This function builds the raw rgba float array from the ramped
// colormap specified in the constructor
void
ColorMap::build_rgba_from_ramp()
{
  for (unsigned int i = 0; i < resolution_; i++)
  {
    const float t = i / (resolution_-1.0);
    vector<float>::iterator loc;
    
    Color diffuse;
    loc = std::lower_bound(rawRampColorT_.begin(), rawRampColorT_.end(), t);
    if (loc != rawRampColorT_.begin())
    {
      const unsigned int index = loc - rawRampColorT_.begin();
      const double d = (t - rawRampColorT_[index-1]) /
	(rawRampColorT_[index] - rawRampColorT_[index-1]);
      diffuse = rawRampColor_[index-1] * (1.0 - d) + rawRampColor_[index] * d;
    }
    else
    {
      diffuse = rawRampColor_.front();
    }

    float alpha = 0.0;
    loc = std::lower_bound(rawRampAlphaT_.begin(), rawRampAlphaT_.end(), t);
    if (loc != rawRampAlphaT_.begin())
    {
      const unsigned int index = loc - rawRampAlphaT_.begin();
      const double d = (t - rawRampAlphaT_[index-1]) /
	(rawRampAlphaT_[index] - rawRampAlphaT_[index-1]);
      alpha = rawRampAlpha_[index-1] * (1.0 - d) + rawRampAlpha_[index] * d;
    }
    else
    {
      alpha = rawRampAlpha_.front();
    }

    rawRGBA_[i*4+0] = diffuse.r();
    rawRGBA_[i*4+1] = diffuse.g();
    rawRGBA_[i*4+2] = diffuse.b();
    rawRGBA_[i*4+3] = alpha;
  }

  // Pad out rawRGBA_ to texture size.
  for (unsigned int i = resolution_; i < 256; i++)
  {
    rawRGBA_[i*4+0] = rawRGBA_[(resolution_-1)*4+0];
    rawRGBA_[i*4+1] = rawRGBA_[(resolution_-1)*4+1];
    rawRGBA_[i*4+2] = rawRGBA_[(resolution_-1)*4+2];
    rawRGBA_[i*4+3] = rawRGBA_[(resolution_-1)*4+3];
  }  
}



// This builds a ramp with regular intervals from the rgba
// array
void
ColorMap::build_ramp_from_rgba()
{
  const int size = 256;
  resolution_ = size;
  rawRampAlpha_.resize(size);
  rawRampAlphaT_.resize(size);  
  rawRampColor_.resize(size);
  rawRampColorT_.resize(size);

  for (int i = 0; i < size; i++)
  {
    const float t = i / (size-1.0);
    rawRampAlphaT_[i] = t;
    rawRampColorT_[i] = t;
    rawRampColor_[i] = Color(rawRGBA_[i*4], rawRGBA_[i*4+1], rawRGBA_[i*4+2]);
    rawRampAlpha_[i] = rawRGBA_[i*4+3];
  }
}


void
ColorMap::build_materials_from_rgba() {
  const int size = 256;
  materials_.resize(size);
  const Color ambient(0.0, 0.0, 0.0);
  const Color specular(0.7, 0.7, 0.7);
  for (int i = 0; i < size; i++)
  {
    Color diffuse(rawRGBA_[i*4], rawRGBA_[i*4+1], rawRGBA_[i*4+2]);
    materials_[i] = new Material(ambient, diffuse, specular, 10);
    materials_[i]->transparency = rawRGBA_[i*4+3];
  }
}


const Color &
ColorMap::getColor(double t)
{
  return materials_[int(t*(materials_.size()-1))]->diffuse;
}

double
ColorMap::getAlpha(double t)
{
  return materials_[int(t*(materials_.size()-1))]->transparency;
}

#define COLORMAP_VERSION 6

void
ColorMap::io(Piostream& stream)
{

  int version= stream.begin_class("ColorMap", COLORMAP_VERSION);

  if ( version > 5)
  {
    PropertyManager::io(stream);
  }

  if ( version > 4)
  {
    Pio(stream, resolution_);
    Pio(stream, min_);
    Pio(stream, max_);
  }
  else if (version == 4 && stream.reading())
  {
    // Bad guess, better than nothing.
    resolution_ = 256;
  }
  Pio(stream, materials_);
  if ( version > 2 )
    Pio(stream, units_);
  if ( version > 1 )
  {
    Pio(stream,rawRampAlpha_);
    Pio(stream,rawRampAlphaT_);
    Pio(stream,rawRampColor_);
    Pio(stream,rawRampColorT_);
      
    if ( stream.reading() )
    {
      build_rgba_from_ramp();
      build_materials_from_rgba();
    }
  }
  stream.end_class();
}


const MaterialHandle&
ColorMap::lookup(double value) const
{
  int idx = int((materials_.size()-1)*(value-min_)/(max_-min_));
  idx = Clamp(idx, 0, (int)materials_.size()-1);
  return materials_[idx];
}


const MaterialHandle&
ColorMap::lookup2(double nvalue) const
{
  int idx = int((materials_.size()-1)*nvalue);
  idx = Clamp(idx, 0, (int)materials_.size()-1);
  return materials_[idx];
}


ColorMap *
ColorMap::create_pseudo_random(int mult)
{
  float rgba[256*4];
  if (mult == 0) mult = 65537; 
  unsigned int seed = 1;
  for (int i = 0; i < 256*4; ++i) {
    seed = seed * mult;
    switch (i%4) {
    case 0: /* Red   */ rgba[i] = (seed % 7)  / 6.0; break;
    case 1: /* Green */ rgba[i] = (seed % 11) / 10.0; break;
    case 2: /* Blue  */ rgba[i] = (seed % 17) / 16.0; break;
    default:
    case 3: /* Alpha */ rgba[i] = 1.0; break;
    }
  }
  rgba[3] = 0.0;

  return new ColorMap(rgba);
}


ColorMap *
ColorMap::create_greyscale()
{
  float rgba[256*4];
  for (int c = 0; c < 256*4; ++c) {
    rgba[c] = (c % 4 == 3) ? 1.0f : (c/4) / 255.0f;
  }

  // Sets the alpha of black to be 0
  // Thus making the minimum value in the colormap transparent
  rgba[3] = 0.0;

  return new ColorMap(rgba);
}

} // End namespace SCIRun


