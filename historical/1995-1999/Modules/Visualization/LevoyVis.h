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
#define TABLE_SIZE 2000

class Levoy;
class LevoyS;

// the type of fnc corresponding to the various cast ray fncs

typedef Color (Levoy::*CRF) ( Vector&, const Point& );

typedef Color (LevoyS::*CRFS) ( const Point&, Vector&,
			     const Point&, const Point&, const int&,
			     const int& );

class Levoy
{
protected:

  // the scalar field
  ScalarFieldRG *homeSFRGrid;

  // the original ray from the eye point to the lookat point
  Vector homeRay;

  // original eye position
  Point eye;

  // Transfer map: the 4 SV-{opacity, r,g,b} arrays
//  double *SVOpacity[4];
  
  double SVOpacity[TABLE_SIZE];
  double SVR[TABLE_SIZE];
  double SVG[TABLE_SIZE];
  double SVB[TABLE_SIZE];

  // appropriately scales the SV for array access
  double SVMultiplier;
  
  // minimum SValue
  double SVmin;

  // the length of a single step along the ray
  double rayStep;

  // diagonal x,y,z components
  double diagx, diagy, diagz;

  // number of voxels in x,y,z directions
  int nx, ny, nz;

  // flag set to true when r,g,b = 0,0,0 throughout the SValues
  int whiteFlag;

  // the bounding box
  BBox box;

  // the ray increments in the u,v direction that are added
  // to the original home ray or the original eye point.
  Vector rayIncrementU, rayIncrementV;

  // raster size
  double xres, yres;

  // background color
  Color backgroundColor;
  
  // the volume rendering routine to be used
  CRF CastRay;

  /*
  */
  
  double DetermineRayStepSize ( int steps );

  void CalculateRayIncrements ( ExtendedView myview,
			       Vector& rayIncrementU, Vector& rayIncrementV );
  
  // associates an opacity or color value (0-1 range) given
  // a scalar value
  inline double AssociateValueO( double scalarValue );
  inline double AssociateValueR( double scalarValue );
  inline double AssociateValueG( double scalarValue );
  inline double AssociateValueB( double scalarValue );

  void PrepareAssociateValue( double sv_min, Array1<double> * ScalarVals,
	      Array1<double> * AssociatedVals );

public:

  Array2<CharColor> Image;

  // constructor
  Levoy();
  Levoy( ScalarFieldRG * grid, Array1<double> * Xarr, Array1<double> * Yarr );

  // destructor
  ~Levoy();

  // set up variables given view info and the stepsize
  void SetUp ( ScalarFieldRG * grid, Array1<double> *Xarr,
	      Array1<double> * Yarr, const ExtendedView& myview,
	      int stepsize );
  void SetUp ( const ExtendedView& myview, int stepsize );

  // uses the new transfer map and ray-bbox intersections
  Color Eight ( Vector& step, const Point& beg  );
  

  Color Nine ( Vector& step, const Point& beg );

  // traces all the rays
  void TraceRays ( int projectionType );


  // Traces rays starting at x=from until x=till
  virtual void PerspectiveTrace( int from, int till );
  
};

/**************************************************************************
 *
 * Associates a value for the particular scalar value
 * given an array of scalar values and corresponding opacity/color values.
 *
 **************************************************************************/


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
  
  LevoyS( ScalarFieldRG * grid,
	 Array1<double> * Xarr, Array1<double> * Yarr );

  // destructor

  ~LevoyS();

  // allows the Salmon image and my volvis stuff to be superimposed

  virtual
    void SetUp ( GeometryData *, int stepsize );

  // uses Salmon view, and Depth and color buffer information
  
  Color Eleven ( const Point& eye, Vector& step,
		const Point& beg , const Point& end, const int& px,
	       const int& py );

  Color Twelve ( const Point& eye, Vector& step,
		const Point& beg , const Point& end, const int& px,
	       const int& py );
  
  // performs a perspective trace in vertical slices
  
  virtual void PerspectiveTrace( int from, int till );
};

void ContinuousRedraw();

#endif
