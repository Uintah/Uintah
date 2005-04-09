#ifndef _RENDERRG_H_
#define _RENDERRG_H_

/*
 *  Render.h: Routines that use the shear-warp VolPack library to
 *            volume render regularly gridded data.
 *
 *  Written by:
 *   Aleksandra Kuswik
 *   Department of Computer Science
 *   University of Utah
 *   May 1997
 *
 *  Copyright (C) 1997 SCI Group
 */


#define VP__NUM_FIELDS            3
#define VP__NUM_SHADE_FIELDS      1
#define VP__NUM_CLASSIFY_FIELDS   2

#define VP__NORMAL_FIELD          0
#define VP__NORMAL_MAX            VP_NORM_MAX
#define VP__SCALAR_FIELD          1
#define VP__SCALAR_MAX            255
#define VP__GRAD_FIELD            2
#define VP__GRAD_MAX              VP_GRAD_MAX

#define VP__COLOR_CHANNELS        3
#define VP__NUM_MATERIALS         1

#include <Classlib/Array2.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Geom/Color.h>
#include <Geom/View.h>
#include <VolPack/volpack.h>

typedef unsigned char Scalar;
typedef unsigned short Normal;
typedef unsigned char Gradient;
typedef struct
{
  Normal normal;
  Scalar scalar;
  Gradient gradient;
} Voxel;

class RenderRGVolume
{
private:
  vpContext *vpc;             // rendering context
  
public:
  RenderRGVolume();
  void NewScalarField( ScalarFieldRG *sfield, View& v );
  void Process( ScalarFieldRG *sfield, ColorMapHandle cmap,
	       View& v, int raster, unsigned char * a );
};

#endif
