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
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ColormapPort.h>

#include <Geom/Color.h>
#include <Geom/View.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Trig.h>

#include <Malloc/Allocator.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>

#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

class Levoy
{
public:
  
  ScalarFieldRG *homeSFRGrid;
  int rasterX, rasterY;

  int colormapFlag;
  ColormapHandle cmap;

  int minSV, maxSV;

  
  Levoy( ScalarFieldRG * grid );

  void AssignRaster ( int x, int y );

void CalculateRayIncrements ( View myview, Vector& rayIncrementU,
			     Vector& rayIncrementV );
     
double DetermineRayStepSize ();

double DetermineFarthestDistance ( const Point& e );

Color& CastRay ( Point eye, Vector ray, double rayStep,	double dmax, Color& backgroundColor, double Xval[], double Yval[] );
		
Color& SecondTry ( Point eye, Vector ray, double rayStep, double dmax, Color& backgroundColor );

Color& Three ( Point eye, Vector ray, double rayStep, double dmax, Color& backgroundColor, double Xval[], double Yval[] );

void TraceRays ( View myview,
		Array2<char>& image, Array2<CharColor>& Image,
		Color& bgColor, double Xval[], double Yval[] );

};

#endif
