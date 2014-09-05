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
#include <Core/Containers/LockingHandle.h>
#include <Core/Geom/Material.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomColormapInterface.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

/* changed to work less stupid with transfer functions
 * Peter-Pike Sloan
 */

namespace SCIRun {

using std::vector;

class SCICORESHARE ColorMap : public Datatype, public GeomColormapInterface {
public:
  vector<float>			rawRampAlpha_;
  vector<float>			rawRampAlphaT_;
  vector<Color>			rawRampColor_;
  vector<float>			rawRampColorT_;
  unsigned char*		rawRGBA_;
  string			units;

private:


  double			min_;
  double			max_;

  vector<MaterialHandle>	colors_;

  bool				scaled_p_;
  bool				blend_p_;

public:

  ColorMap();
  ColorMap(const ColorMap&);
  ColorMap(int nlevels, double min, double max);

  ColorMap(const vector<Color>& rgb,
	   const vector<float>& rgbT,
	   const vector<float>& alphas,
	   const vector<float>& alphaT);
  virtual ~ColorMap();
  virtual ColorMap*	clone();

  void			SetRaw(const vector<Color>& rgb,
			       const vector<float>& rgbT,
			       const vector<float>& alphas,
			       const vector<float>& alphaT);

  void			Build1d(const int size=256);

  // are the colors scaled to some data?
  bool			IsScaled(){ return scaled_p_;} 

  void			Scale(double newmin, double newmax)
  { min_ = newmin; max_ = newmax; scaled_p_ = true;}

  void			ResetScale() 
  { min_ = -1; max_ = 1; scaled_p_ = false; }

  // return the number of color points in the colormap
  int			size() const { return rawRampColor_.size();}
 
  // Lookup which color value would be associated with in the colormap.
  MaterialHandle&	lookup(double value);

  // Lookup a color in the colormap by a value that has already been fitted
  // to the min/max of the colormap (value should be between 1 and ncolors).
  MaterialHandle&	lookup2(double value);

  virtual double	getMin() const;
  virtual double	getMax() const;
  Color			getColor(double t);
  double		getAlpha(double t);

  bool			blend_p() { return blend_p_; }
  void			set_blend(bool);

  // Persistent representation...
  virtual void		io(Piostream&);
  static PersistentTypeID type_id;
};

typedef LockingHandle<ColorMap> ColorMapHandle;


} // End namespace SCIRun


#endif
