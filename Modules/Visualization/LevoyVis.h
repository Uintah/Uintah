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
#include <Geometry/Plane.h>
#include <Geometry/BBox.h>

#include <Math/Trig.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <stdlib.h>

#define CANVAS_WIDTH 200

class Levoy;

// the type of fnc corresponding to the various cast ray fncs

typedef Color (Levoy::*CRF) ( const Point&, Vector&,
			     const Point&, const Point&, const int&,
			     const int& );

// the type of fnc corresponding to Perspective/Parallel trace fncs

typedef void (Levoy::*TR) ( int, int );

class Levoy
{
private:
  
  ScalarFieldRG *homeSFRGrid;

  int colormapFlag;
  ColormapHandle cmap;

  Color backgroundColor;

  Array1<double> * AssociatedVals;
  Array1<double> * ScalarVals;

  double rayStep;
  double dmax, dmin;

  int debug_flag;

  int whiteFlag;

  BBox box;
  Point gmin, gmax;

  // the ray increments in the u,v direction that are added
  // to the original home ray or the original eye point.
  
  Vector rayIncrementU, rayIncrementV;

  // original eye position

  Point eye;
  
  // the original ray from the eye point to the lookat point

  Vector homeRay;

  // consts used in determination of the distance between
  // 2 clipping planes

  double clipConst;

  double nearTohome;

  // pointer to the array of char colors

  CRF CastRay;

  // the size of the image

  int x, y;


  // the depth and color buffers

  Array2<double>* DepthBuffer;
  Array2<Color>*  ColorBuffer;

  int CombineSalmonFlag;

public:

  Array2<CharColor> * Image;

  // constructor
  
  Levoy( ScalarFieldRG * grid, ColormapIPort * c,
	Color& bg, Array1<double> * Xarr, Array1<double> * Yarr );

  ~Levoy();

  void CalculateRayIncrements ( View myview, Vector& rayIncrementU,
			       Vector& rayIncrementV, const int& rasterX,
			       const int& rasterY );
  
  double DetermineRayStepSize ();

  double DetermineFarthestDistance ( const Point& e );
  
  Color Five ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  Color Six ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  Color Seven ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  
  // uses the new transfer map

  Color Eight ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end, const int& px,
	       const int& py );

  Color Nine ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end, const int& px,
	       const int& py );

  Color Ten ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end, const int& px,
	       const int& py );

  
  // uses Salmon view stuff
  
  Color Eleven ( const Point& eye, Vector& step,
		const Point& beg , const Point& end, const int& px,
	       const int& py );

  Color Twelve ( const Point& eye, Vector& step,
		const Point& beg , const Point& end, const int& px,
	       const int& py );
  
  Color Thirteen ( const Point& eye, Vector& step,
		  const Point& beg , const Point& end, const int& px,
	       const int& py );

  
  Array2<CharColor>* TraceRays ( int projectionType );
  

  // no depth buffer or color buffer info; therefore, Salmon view
  // is not applicable

  void SetUp ( const View& myview, const int& x, const int& y );

  // allows the Salmon image and my volvis stuff to be superimposed

  void SetUp ( const View& myview, const int& x, const int& y,
	      Array2<double>* db, Array2<Color>* cb, double Cnear,
	      double Cfar );

  
  void PerspectiveTrace( int from, int till );
  void ParallelTrace( int from, int till );

  double AssociateValue( double scalarValue,
			const Array1<double>& SVA, const Array1<double>& OA );
};

#endif
