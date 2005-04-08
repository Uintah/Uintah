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

#ifndef SCI_project_ColorMap_h
#define SCI_project_ColorMap_h 1

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geom/Material.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomColormapInterface.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Geom/share.h>

/* changed to work less stupid with transfer functions
 * Peter-Pike Sloan
 */

namespace SCIRun {

using std::vector;

class ColorMap : public PropertyManager,
                              public GeomColormapInterface
{
public:
  ColorMap(const ColorMap&);

  ColorMap(const vector<Color>& rgb,
	   const vector<float>& rgbT,
	   const vector<float>& alphas,
	   const vector<float>& alphaT,
	   unsigned int resolution = 256);
  virtual ~ColorMap();
  virtual ColorMap*	clone();


  const vector<Color> &get_rgbs() { return rawRampColor_; }
  const vector<float> &get_rgbT() { return rawRampColorT_; }
  const vector<float> &get_alphas() { return rawRampAlpha_; }
  const vector<float> &get_alphaT() { return rawRampAlphaT_; }

  // Functions for handling the color scale of the data.
  bool                  IsScaled() { return is_scaled_; }
  void			Scale(double newmin, double newmax)
  { min_ = newmin; max_ = newmax; is_scaled_ = true; }
  void			ResetScale() 
  { min_ = -1.0; max_ = 1.0; is_scaled_ = false; }
  virtual double	getMin() const;
  virtual double	getMax() const;

  void                  set_units(const string &u) { units_ = u; }
  string                units() { return units_; }

  // Lookup which color value would be associated with in the colormap.
  const MaterialHandle&	lookup(double value) const;

  // Lookup a color in the colormap by a value that has already been fitted
  // to the min/max of the colormap (value should be between 1 and ncolors).
  const MaterialHandle&	lookup2(double value) const;

  const Color &		getColor(double t);
  double		getAlpha(double t);

  const float *         get_rgba() { return rawRGBA_; }
  unsigned int          resolution() { return resolution_; }

public:

  // Persistent representation.
  virtual void		io(Piostream&);
  static PersistentTypeID type_id;

private:
  friend class GeomColorMap;

  ColorMap();
  void			Build1d();

  static Persistent *maker();

  vector<float>			rawRampAlpha_;
  vector<float>			rawRampAlphaT_;
  vector<Color>			rawRampColor_;
  vector<float>			rawRampColorT_;

  unsigned int                  resolution_;
  float*         		rawRGBA_;
  string			units_;

  double			min_;
  double			max_;
  bool                          is_scaled_;

  vector<MaterialHandle>	colors_;
};


typedef LockingHandle<ColorMap> ColorMapHandle;


} // End namespace SCIRun


#endif
