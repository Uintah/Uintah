/*
 *  LevoyVis.h        Definitions of Mark Levoy's volume visualization
 *                    on structured grids.
 *
 *  Written by:
 *   Aleksandra Kuswik
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_project_LevoyVol_h
#define SCI_project_LevoyVol_h 1

#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>

#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ColormapPort.h>

#include <Geom/Color.h>
#include <Geom/View.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

#include <Math/Trig.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <stdlib.h>

#define CANVAS_WIDTH 201

const Color BLACK( 0., 0., 0. );
const Color WHITE( 1., 1., 1. );

class Levoy
{
private:
  
  ScalarFieldRG *homeSFRGrid;

  int colormapFlag;
  ColormapHandle cmap;

  Color backgroundColor;

//  double * Opacity;
//  double * SVArray;

  Array1<double> Opacity;
  Array1<double> SVArray;

  double dmax;
  
public:

  // constructor
  
  Levoy( ScalarFieldRG * grid, ColormapIPort * c,
	Color& bg, Array1<double> Xarr, Array1<double> Yarr );

  void CalculateRayIncrements ( View myview, Vector& rayIncrementU,
			       Vector& rayIncrementV, int rasterX, int rasterY );
  
  double DetermineRayStepSize ();

  double DetermineFarthestDistance ( const Point& e );

  Color& CastRay ( Point eye, Vector ray, double rayStep );
  
  Color& Three ( Point eye, Vector ray, double rayStep );

  Color& Four ( Point eye, Vector ray, double rayStep );

  Color& Five ( Point eye, Vector ray, double rayStep );

  Array2<CharColor>* TraceRays ( View myview, int x, int y,
				int projectionType );


};

#endif
