/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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


ColorMap::ColorMap()
  : rawRampAlpha_(),
    rawRampAlphaT_(),
    rawRampColor_(),
    rawRampColorT_(),
    resolution_(0),
    rawRGBA_(0),
    units_(),
    min_(-1.0),
    max_(1.0),
    is_scaled_(false),
    colors_()
{
}


ColorMap::ColorMap(const ColorMap& copy)
  : rawRampAlpha_(copy.rawRampAlpha_),
    rawRampAlphaT_(copy.rawRampAlphaT_),
    rawRampColor_(copy.rawRampColor_),
    rawRampColorT_(copy.rawRampColorT_),
    resolution_(copy.resolution_),
    rawRGBA_(0),
    min_(copy.min_),
    max_(copy.max_),
    is_scaled_(copy.is_scaled_),
    colors_()
{
  Build1d();
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
   rawRGBA_(0),
   min_(-1.0),
   max_(1.0),
   is_scaled_(false),
   colors_()
{
  Build1d();
}


ColorMap::~ColorMap()
{
  if (rawRGBA_)
    delete[] rawRGBA_;
  
  colors_.clear();
}


ColorMap* ColorMap::clone()
{
  return scinew ColorMap(*this);
}


void
ColorMap::Build1d()
{
  unsigned int i;
  const unsigned int size = resolution_;

  if (rawRGBA_) { delete[] rawRGBA_; }
  rawRGBA_ = scinew float[256 * 4];

  colors_.resize(size);
  
  const Color ambient(0.0, 0.0, 0.0);
  const Color specular(0.7, 0.7, 0.7);

  for (i = 0; i < size; i++)
  {
    const float t = i / (size-1.0);
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

    colors_[i] = scinew Material(ambient, diffuse, specular, 10);
    colors_[i]->transparency = alpha;
  }

  // Pad out rawRGBA_ to texture size.
  for (i = size; i < 256; i++)
  {
    rawRGBA_[i*4+0] = rawRGBA_[(size-1)*4+0];
    rawRGBA_[i*4+1] = rawRGBA_[(size-1)*4+1];
    rawRGBA_[i*4+2] = rawRGBA_[(size-1)*4+2];
    rawRGBA_[i*4+3] = rawRGBA_[(size-1)*4+3];
  }  
}


const Color &
ColorMap::getColor(double t)
{
  return colors_[int(t*(colors_.size()-1))]->diffuse;
}

double
ColorMap::getAlpha(double t)
{
  return colors_[int(t*(colors_.size()-1))]->transparency;
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
  Pio(stream, colors_);
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
      Build1d();
    }
  }
  stream.end_class();
}


const MaterialHandle&
ColorMap::lookup(double value) const
{
  int idx=int((colors_.size()-1)*(value-min_)/(max_-min_));
  if(idx<0)
    idx=0;
  else if(idx > (int)colors_.size()-1)
    idx=colors_.size()-1;
  return colors_[idx];
}


const MaterialHandle&
ColorMap::lookup2(double nvalue) const
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


