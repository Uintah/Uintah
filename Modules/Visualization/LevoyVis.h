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
#include <Datatypes/Image.h>
#include <Datatypes/GeometryPort.h>

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


#define VIEW_PORT_SIZE 600

#define CANVAS_WIDTH 200

class Levoy;
class LevoyS;

// the type of fnc corresponding to the various cast ray fncs

typedef Color (Levoy::*CRF) ( const Point&, Vector&,
			     const Point&, const Point& );

typedef Color (LevoyS::*CRFS) ( const Point&, Vector&,
			     const Point&, const Point&, const int&,
			     const int& );

// the type of fnc corresponding to Perspective/Parallel trace fncs

typedef void (Levoy::*TR) ( int, int );

class Levoy
{
protected:
  
  ScalarFieldRG *homeSFRGrid;

  int colormapFlag;
  ColormapHandle cmap;

  Color backgroundColor;

  Array1<double> * AssociatedVals;
  Array1<double> * ScalarVals;


  Array1<double> * Slopes;

  double rayStep;
  double dmax, dmin;

  int debug_flag;

  int whiteFlag;

  BBox box;
  Point gmin, gmax;

  // the ray increments in the u,v direction that are added
  // to the original home ray or the original eye point.
  
  Vector rayIncrementU, rayIncrementV;

  // L vector is the vector pointing toward the sun

  Vector Lvector;

  // lighting coefficients

  double ambient_coeff, diffuse_coeff, specular_coeff, specular_iter;

  // original eye position

  Point eye;
  
  // the original ray from the eye point to the lookat point

  Vector homeRay;

  // pointer to the array of char colors

  CRF CastRay;

  // the size of the image

  int x, y;

  // flag for centering

  int centerFlag;

  // offsets from the center
  
  int centerOffsetU, centerOffsetV;
  

  double DetermineRayStepSize ();

  void CalculateRayIncrements ( View myview, Vector& rayIncrementU,
			       Vector& rayIncrementV, const int& rasterX,
			       const int& rasterY );
  
  double DetermineFarthestDistance ( const Point& e );

  // associates an opacity or color value (0-1 range) given
  // a scalar value

  double AssociateValue( double scalarValue,
			const Array1<double>& SVA, const Array1<double>& OA );
  
  double AssociateValue( double scalarValue,
			const Array1<double>& SVA, const Array1<double>& OA,
			const Array1<double>& Sl );
  
  double LightingComponent( const Vector& normal );
public:

  Array2<CharColor> * Image;

  // constructor
  
  Levoy( ScalarFieldRG * grid, ColormapIPort * c,
	Color& bg, Array1<double> * Xarr, Array1<double> * Yarr );

  // destructor
  
  ~Levoy();

  // no depth buffer or color buffer info; therefore, Salmon view
  // is not applicable

  void SetUp ( const View& myview, int x, int y );

  virtual void SetUp ( GeometryData * g );
  
  // initial, most brute force algorithm

  Color Five ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  Color Six ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  Color Seven ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  
  // uses the new transfer map and ray-bbox intersections

  Color Eight ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  Color Nine ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  Color Ten ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );


  // uses the Bresenham algorithm and no interpolation
  
  Color Bresenham1 ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  Color Bresenham2 ( const Point& eye, Vector& step,
	      const Point& beg , const Point& end );

  // traces all the rays
  
  Array2<CharColor>* TraceRays ( int projectionType );

  // traces rays repeat number of times, and takes the time it took
  
  Array2<CharColor>* BatchTrace ( int projectionType, int repeat );
  

  // Traces rays starting at x=from until x=till
  
  virtual void PerspectiveTrace( int from, int till );
  
  void ParallelTrace( int from, int till );
};

/**************************************************************************
 *
 * Associates a value for the particular scalar value
 * given an array of scalar values and corresponding opacity/color values.
 *
 **************************************************************************/

inline
double
Levoy::AssociateValue ( double scalarValue,
		       const Array1<double>& SVA, const Array1<double>& OA,
		       const Array1<double>& Sl )
{
  double opacity = -1;
  int i;
  
  for ( i = 1; i < SVA.size(); i++ )
    if ( scalarValue <= SVA[i] )
      {
	opacity = Sl[i] * ( scalarValue - SVA[i-1] ) + OA[i-1];
	break;
      }

  ASSERT ( opacity != -1 );

  return opacity;

}


class LevoyS : public Levoy
{

private:
  
  // consts used in determination of the distance between
  // 2 clipping planes

  double clipConst;

  double nearTohome;

  // the depth and color buffers

  Array2<double>* DepthBuffer;
  Array2<Color>*  BackgroundBuffer;

  CRFS CastRay;

  GeometryData *geom;

public:

  // constructor
  
  LevoyS( ScalarFieldRG * grid, ColormapIPort * c,
	Color& bg, Array1<double> * Xarr, Array1<double> * Yarr );

  // destructor

  ~LevoyS();

  // allows the Salmon image and my volvis stuff to be superimposed

#if 0  
  virtual
    void SetUp ( const View& myview, int x, int y,
	      Array2<double>* db, Array2<Color>* cb, double Cnear,
	      double Cfar );
#endif
  
  virtual
    void SetUp ( GeometryData * );

  // uses Salmon view, and Depth and color buffer information
  
  Color Eleven ( const Point& eye, Vector& step,
		const Point& beg , const Point& end, const int& px,
	       const int& py );

  Color Twelve ( const Point& eye, Vector& step,
		const Point& beg , const Point& end, const int& px,
	       const int& py );
  
  Color Thirteen ( const Point& eye, Vector& step,
		  const Point& beg , const Point& end, const int& px,
	       const int& py );

  // performs a perspective trace in vertical slices
  
  virtual void PerspectiveTrace( int from, int till );
};

  
#endif
