/*
 *  LevoyVis.cc       Implementation of Mark Levoy's volume visualization
 *                    on structured grids.
 *
 *  Written by:
 *   Aleksandra Kuswik
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 *
 *
 * IMPORTANT: if i want the image to allign with the salmon view
 * if the salmon raster exceeds that of VIEW_PORT_SIZE, i must
 * change one thing: when i loop through the rays, i must
 * start at the very corner ray, and go on.  otherwise, the rays
 * get split evenly. (i'm sure this doesn't make sense to anyone
 * but me).
 *
 */

#include <config.h>
#undef SCI_ASSERTION_LEVEL_3
#define SCI_ASSERTION_LEVEL_2

#include "kuswik.h"
#include <Classlib/Timer.h>
#include <Modules/Visualization/LevoyVis.h>
#include <Math/MiscMath.h>
#include "FastRender.h"


/* the tiny epsilon vector added to counteract possible
   calculation error */

Vector EpsilonVector( 1.e-6, 1.e-6, 1.e-6 );

/* the most tiny opacity contribution of a point in
   the volume */

double  EpsilonContribution = 0.01;
double  EpsilonAlpha = log( 0.01 );



inline double
Levoy::AssociateValueO ( double scalarValue )
{
  return(SVOpacity[ int( (scalarValue - SVmin) * SVMultiplier) ] );
}

inline double
Levoy::AssociateValueR ( double scalarValue )
{
  return(SVR[ int( (scalarValue - SVmin) * SVMultiplier) ] );
}

inline double
Levoy::AssociateValueG ( double scalarValue )
{
  return(SVG[ int( (scalarValue - SVmin) * SVMultiplier) ] );
}

inline double
Levoy::AssociateValueB ( double scalarValue )
{
  return(SVB[ int( (scalarValue - SVmin) * SVMultiplier) ] );
}


void
Levoy::PrepareAssociateValue( double sv_max, Array1<double> * ScalarVals,
			     Array1<double> * AssociatedVals )
{
  int i, j, k;
  int beg, end;
  int counter;
  Array1<double> Slopes[4];

  // calculate slopes between consecutive opacity values
  
  for ( i = 0; i < 4; i++ )
    Slopes[i].add( 0. );

  for ( i = 0; i < 4; i++ )
    for ( j = 1; j < ScalarVals[i].size(); j++ )
      Slopes[i].add( ( AssociatedVals[i][j] - AssociatedVals[i][j-1] ) /
		    ( ScalarVals[i][j] - ScalarVals[i][j-1] ) );

  // scalar value multiplier which scales the SV appropriately
  // for array access

  SVMultiplier = ( TABLE_SIZE - 1 ) / ( sv_max - SVmin );

  j = 0;
  counter = 0;
  SVOpacity[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVOpacity[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
				SVMultiplier + AssociatedVals[j][i] );
    }


  j = 1;
  counter = 0;
  SVR[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVR[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
			  SVMultiplier + AssociatedVals[j][i] );
    }


  j = 2;
  counter = 0;
  SVG[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVG[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
			  SVMultiplier + AssociatedVals[j][i] );
    }


  j = 3;
  counter = 0;
  SVB[counter++] = AssociatedVals[j][0];
  
  for ( i = 0; i < ScalarVals[j].size() - 1; i++ )
    {
      beg = int( ScalarVals[j][i] * SVMultiplier ) + 1;
      end = int( ScalarVals[j][i+1] * SVMultiplier );
      
      for( k = beg; k <= end; k++ )
	SVB[counter++] = ( Slopes[j][i+1] * ( k - beg ) /
			  SVMultiplier + AssociatedVals[j][i] );
    }


  
  if ( counter > TABLE_SIZE )
    {
      cerr << "NOT GOOD AT ALL\n";
      ASSERT( counter == TABLE_SIZE );
    }

#if 0  
  for ( j = 0; j < 4; j++ )
    if ( TABLE_SIZE != SVOpacity[j].size() )
      cerr << "TABLE SIZE is: " << TABLE_SIZE << " ; size is: " << SVOpacity[j].size() << endl;

  
  for ( j = 0; j < 4; j++ )
    {
      for ( i = 0; i < SVOpacity[j].size(); i++ )
	{
	  if ( i%20 == 0 )
	    cerr << endl;

	  cerr << SVOpacity[j][i] << "  ";
	}
      cerr << endl;
    }
#endif
  

  // if the rgb are at 1 for every scalar value, set the
  // whiteFlag to be true, otherwise it is false

  whiteFlag = TRUE;
  
  for ( i = 1; i < 4; i++ )
    {
      if ( AssociatedVals[i].size() != 2 )
	{
	  whiteFlag = FALSE;
	  break;
	}
      if ( AssociatedVals[i][0] != 1. || AssociatedVals[i][1] != 1. )
	{
	  whiteFlag = FALSE;
	  break;
	}
    }
}


/**************************************************************************
 *
 * Constructor
 *
 **************************************************************************/

Levoy::Levoy ()
{
  homeSFRGrid = NULL;
}

Levoy::Levoy ( ScalarFieldRG * grid, Array1<double> * ScalarVals,
	      Array1<double> * AssociatedVals )
{
  Point bmin, bmax;
  double sv_max;
  
  homeSFRGrid = grid;
  homeSFRGrid->get_bounds( bmin, bmax );
  homeSFRGrid->get_minmax( SVmin, sv_max );

  Vector diagonal=bmax-bmin;
  diagx=diagonal.x();
  diagy=diagonal.y();
  diagz=diagonal.z();

  nx=homeSFRGrid->nx;
  ny=homeSFRGrid->ny;
  nz=homeSFRGrid->nz;
  
  // set up the bounding box for the image
  box.extend( bmin );
  box.extend( bmax );

  PrepareAssociateValue( sv_max, ScalarVals, AssociatedVals );
}


void
Levoy::SetUp ( ScalarFieldRG * grid, Array1<double> * ScalarVals,
	      Array1<double> * AssociatedVals, const ExtendedView& myview,
	      int stepsize )
{
  Point bmin, bmax;
  double sv_max;
  
  homeSFRGrid = grid;
  homeSFRGrid->get_bounds( bmin, bmax );
  homeSFRGrid->get_minmax( SVmin, sv_max );

  Vector diagonal=bmax-bmin;
  diagx=diagonal.x();
  diagy=diagonal.y();
  diagz=diagonal.z();

  nx=homeSFRGrid->nx;
  ny=homeSFRGrid->ny;
  nz=homeSFRGrid->nz;
  
  // set up the bounding box for the image
  box.extend( bmin );
  box.extend( bmax );

  PrepareAssociateValue( sv_max, ScalarVals, AssociatedVals );
  
  // temporary storage for eye position
  eye = myview.eyep();

  // assign background color
  backgroundColor = myview.bg();

  xres = myview.xres();
  yres = myview.yres();
  
  // useful for ray-bbox intersection
  box.PrepareIntersect( eye );

  // the original ray from the eye point to the lookat point
  homeRay = myview.lookat() - eye;
  homeRay.normalize();

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space
  rayStep = DetermineRayStepSize ( stepsize );


  // deallocate old array and allocate enough space
  Image.newsize( myview.yres(), myview.xres() );

  
  // initialize to the default (black) color
  Image.initialize( myview.bg() );


  // calculate the increments to be added in the u,v directions
  CalculateRayIncrements( myview, rayIncrementU, rayIncrementV );

  // determine the function to use
  if ( whiteFlag )
    CastRay = &Levoy::Eight;
  else // user defined SV-color map
    CastRay = &Levoy::Nine;
}
/**************************************************************************
 *
 * destructor
 *
 **************************************************************************/

Levoy::~Levoy()
{
}



/**************************************************************************
 *
 * attempts to determine the most appropriate interval which is used
 * to step through the volume.
 *
 **************************************************************************/

double
Levoy::DetermineRayStepSize ( int steps )
{
  double small[3];
  double result;
  
  // calculate a step size that is about the length of one voxel
  small[0] = ( box.max().x() - box.min().x() ) / nx;
  small[1] = ( box.max().y() - box.min().y() ) / ny;
  small[2] = ( box.max().z() - box.min().z() ) / nz;

  // set rayStep to the smallest of the step sizes
  if ( small[0] < small[1] )
    if ( small[0] < small[2] )
      result = small[0];
    else
      result = small[2];
  else if ( small[1] < small[2] )
    result = small[1];
  else
    result = small[2];

  return( steps * result );
}



/**************************************************************************
 *
 * part of a ray tracer; calculates the ray increments in the u,v directions
 *
 **************************************************************************/

void
Levoy::CalculateRayIncrements ( ExtendedView myview,
			       Vector& rayIncrementU, Vector& rayIncrementV )
{
  myview.get_normalized_viewplane( rayIncrementU, rayIncrementV );

  double aspect = double( myview.xres() ) / double( myview.yres() );
  double fovy=RtoD(2*Atan(aspect*Tan(DtoR(myview.fov()/2.))));
  
  double lengthY = 2 * tan( DtoR( fovy / 2 ) );

  rayIncrementV *= lengthY / myview.yres();
  rayIncrementU *= lengthY / myview.yres();
}



/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
Levoy::SetUp ( const ExtendedView& myview, int stepsize )
{
  // temporary storage for eye position

  eye = myview.eyep();

  // assign background color

  backgroundColor = myview.bg();

  xres = myview.xres();
  yres = myview.yres();
  
  // useful for ray-bbox intersection
  box.PrepareIntersect( eye );

  // the original ray from the eye point to the lookat point

  homeRay = myview.lookat() - eye;
  homeRay.normalize();

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space

  rayStep = DetermineRayStepSize ( stepsize );


  // deallocate old array and allocate enough space

  Image.newsize( myview.yres(), myview.xres() );

  
  // initialize to the default (black) color

  Image.initialize( myview.bg() );

  // calculate the increments to be added in the u,v directions

  CalculateRayIncrements( myview, rayIncrementU, rayIncrementV );

  // determine the function to use

  if ( whiteFlag )
    CastRay = &Levoy::Eight;
  else // user defined SV-color map
    CastRay = &Levoy::Nine;


}



/**************************************************************************
 *
 *
 *
 **************************************************************************/


/**************************************************************************
 *
 * calculates the pixel color by casting a ray starting at the first
 * intersection of the ray with the bbox, and ending at the second
 * intersection of the ray with the bbox.
 *
 * the opacity map is not limited to 5 nodes.
 * opacity varies according with the transfer map specified
 * by the user.  the color associated with any scalar value
 * is fixed to be WHITE.
 *
 **************************************************************************/

		 
Color
Levoy::Eight ( Vector& step, const Point& beg )
{
  double r, g, b;
  step = step * rayStep;
  BasicVolRender( step.x(), step.y(), step.z(), rayStep,
			beg.x(), beg.y(),
			beg.z(), SVOpacity, SVmin, SVMultiplier,
			box.min().x(), box.min().y(), box.min().z(),
			homeSFRGrid->grid.get_dataptr(), backgroundColor.r(),
			backgroundColor.g(), backgroundColor.b(),
			nx, ny, nz, diagx, diagy, diagz, &r,&g,&b );
  return ( Color ( r,g,b) );
}



/**************************************************************************
 *
 * calculates the pixel color by casting a ray starting at the first
 * intersection of the ray with the bbox, and ending at the second
 * intersection of the ray with the bbox.
 *
 * the opacity map is not limited to 5 nodes.
 * opacity and color vary according with the transfer map specified
 * by the user.
 *
 **************************************************************************/


Color
Levoy::Nine ( Vector& step, const Point& beg )
{
  double r, g, b;
  step = step * rayStep;
  ColorVolRender( step.x(), step.y(), step.z(), rayStep,
			beg.x(), beg.y(),
			beg.z(), SVOpacity, SVR, SVG, SVB,
		        SVmin, SVMultiplier,
			box.min().x(), box.min().y(), box.min().z(),
			homeSFRGrid->grid.get_dataptr(), backgroundColor.r(),
			backgroundColor.g(), backgroundColor.b(),
			nx, ny, nz, diagx, diagy, diagz, &r,&g,&b );
  return ( Color ( r,g,b) );
}



/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
Levoy::PerspectiveTrace( int from, int till )
{
  int loop, pool;
  Vector rayToTrace;
  Color pixelColor;
  Point newbeg;

  for ( pool = from; pool < till; pool++ )
    {
      for ( loop = 0; loop < yres; loop++ )
	{
	  if ( loop == 45 && pool == 45 )
	    {
	      cerr << ".";
	    }
	  // assign ray direction depending on the pixel
	  rayToTrace = homeRay;
	  rayToTrace += rayIncrementU * ( pool - xres/2 );
	  rayToTrace += rayIncrementV * ( loop - yres/2 );
	  rayToTrace.normalize();
	  
	  if ( box.Intersect( eye, rayToTrace, newbeg ) )
	    pixelColor = (this->*CastRay)( rayToTrace, newbeg );
	  else
	    pixelColor = backgroundColor;

	  /*
	     i need to do it this way, unless i can write
	     an operator overload fnc for ()=
	     Image( loop, pool ) returns a value; it cannot
	     be assigned.
	     Image( loop, pool ) = pixelColor;
	     */

	  (Image( loop, pool )).red =
	    (char)(pixelColor.r()*255);
	  (Image( loop, pool )).green =
	    (char)(pixelColor.g()*255);
	  (Image( loop, pool )).blue =
	    (char)(pixelColor.b()*255);
	}
    }
}



/**************************************************************************
 *
 * Constructor
 *
 **************************************************************************/

LevoyS::LevoyS ( ScalarFieldRG * grid,
		Array1<double> * Xarr, Array1<double> * Yarr )
: Levoy( grid, Xarr, Yarr )
{
}


/**************************************************************************
 *
 *
 *
 **************************************************************************/

LevoyS::~LevoyS()
{
}



/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
LevoyS::SetUp ( GeometryData * g, int stepsize )
{
  g->Print();
  
  geom = g;
  
  // temporary storage for eye position

  eye = g->view->eyep();

  // the original ray from the eye point to the lookat point

  homeRay = g->view->lookat() - eye;
  homeRay.normalize();

  // the distance between the 2 clipping planes
  
  clipConst = ( g->zfar - g->znear ) / homeRay.length();

  // the ratio of near clipping plane to homeRaylength

  nearTohome = g->znear / homeRay.length();

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space

  rayStep = DetermineRayStepSize ( stepsize );

  // make sure that raster sizes do not exceed the raster size
  // of the OpenGL window.

  int rx, ry;

  rx = g->xres;
  ry = g->yres;

  if ( g->xres > VIEW_PORT_SIZE )
    g->xres = VIEW_PORT_SIZE;

  if ( g->yres > VIEW_PORT_SIZE )
    g->yres = VIEW_PORT_SIZE;

  // create and initialize the array containing the image

  CharColor temp;


  // deallocate, allocate, and initialize the array
  
  Image.newsize( g->yres, g->xres );
  
  Image.initialize( temp );

  // calculate the increments to be added in the u,v directions

  ExtendedView joy ( *(g->view), xres, yres, BLACK );

  CalculateRayIncrements( joy, rayIncrementU, rayIncrementV );

  // determine the function to use

  if ( whiteFlag )
    CastRay = &LevoyS::Eleven;
  else // user defined SV-color map
    CastRay = &LevoyS::Twelve;
}



/**************************************************************************
 *
 * calculates the pixel color by casting a ray starting at the first
 * intersection of the ray with the bbox, and ending at the second
 * intersection of the ray with the bbox or the z-buffer value provided
 * for that particular point in space.
 *
 * the opacity map is not limited to 5 nodes.
 * opacity varies according with the transfer map specified
 * by the user.  the color associated with any scalar value
 * is fixed to be white.
 *
 * the background color is that provided by an array. 
 *
 **************************************************************************/

Color
LevoyS::Eleven ( const Point& eye, Vector& step,
		const Point& beg, const Point& end, const int& px,
		const int& py )
{
  // as the ray passes through the volume, the voxel's
  // color contribution decreases
  
  double contribution;

  // the current location in the volume

  Point atPoint;

  // accumulated color as the ray passes through the volume
  
  Color accumulatedColor;

  // a scalar value corresponds to a particular point
  // in space
  
  double scalarValue;

  // opacity for a particular position in the volume

  double opacity;

  // step length

  double stepLength;

  // the length of the ray to be cast

  double whatever = 1;
  double mylength  = whatever;

  // the length of ray stepped through so far

  double traversed;

  // contribution of the first voxel's color and opacity
  
  contribution = 1.0;
  
  // since no voxels have been crossed, no color has been
  // accumulated.

  accumulatedColor = BLACK;


  int templength = step.length() * ( clipConst *
				    geom->depthbuffer->get_depth(px,py) + nearTohome );

  // compare the 2 lengths, pick the shorter one

  if ( templength < mylength )
    mylength = templength;
  

  // find step vector

  step.normalize();

  step *= rayStep;

  // memorize length

  stepLength = step.length();

  // the color for rendering

  Color colr = WHITE;

  // position of the point in space at the beginning of the cast
  
  atPoint = beg + EpsilonVector - step;

  // the length of the ray to be cast

  mylength = (end - eye).length() - (beg - eye).length();


  // the ray cast begins before entering the volume

  traversed = - stepLength;

  // begin traversing through the volume at the first point of
  // intersection of the ray with the bbox.  keep going through
  // the volume until the contribution of the voxel color
  // is almost none, or until the second intersection point has
  // been reached.

  while ( contribution > EpsilonContribution && traversed <= mylength )
    {
      atPoint += step;
      scalarValue = 0;
      traversed += stepLength;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  opacity = AssociateValueO( scalarValue );

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel.  if contribution is minute, the background
  // is not visible therefore don't add the background color
  // to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = geom->colorbuffer->get_pixel(px,py) * contribution +
      accumulatedColor;

  return accumulatedColor;
}






/**************************************************************************
 *
 * calculates the pixel color by casting a ray starting at the first
 * intersection of the ray with the bbox, and ending at the second
 * intersection of the ray with the bbox.
 *
 * the opacity map is not limited to 5 nodes.
 * opacity and color vary according with the transfer map specified
 * by the user.
 *
 **************************************************************************/


Color
LevoyS::Twelve ( const Point& eye, Vector& step,
		const Point& beg, const Point& end, const int& px,
		const int& py )
{
#if 0
  // as the ray passes through the volume, the voxel's
  // color contribution decreases
  
  double contribution;

  // the current location in the volume

  Point atPoint;

  // accumulated color as the ray passes through the volume
  
  Color accumulatedColor;

  // a scalar value corresponds to a particular point
  // in space
  
  double scalarValue;

  // step length

  double stepLength;

  // the length of the ray to be cast

  double whatever = 1;
  double mylength = whatever;

  // the length of ray stepped through so far

  double traversed;

  // the color corresponding to the scalar value

  Color colr;

  // the opacity corresponding to the scalar value

  double opacity;

  
  // contribution of the first voxel's color and opacity
  
  contribution = 1.0;
  
  // since no voxels have been crossed, no color has been
  // accumulated.

  accumulatedColor = BLACK;

  // calculate the length from the eye to the DepthBuffer based point
  // in space

  int templength = step.length() * ( clipConst *
				    geom->depthbuffer->get_depth(px,py) + nearTohome );

  // compare the 2 lengths, pick the shorter one

  if ( templength < mylength )
    mylength = templength;
  

  // find step vector
  
  step.normalize();

  step *= rayStep;

  // set step length

  stepLength = step.length();

  // position of the point in space at the beginning of the cast
  
  atPoint = beg + EpsilonVector - step;

  // the length of the ray to be cast

  mylength = (end - eye).length() - (beg - eye).length();


  // the ray cast begins before entering the volume

  traversed = - stepLength;

  // begin traversing through the volume at the first point of
  // intersection of the ray with the bbox.  keep going through
  // the volume until the contribution of the voxel color
  // is almost none, or until the second intersection point has
  // been reached.

  while ( contribution > EpsilonContribution && traversed <= mylength )
    {
      atPoint += step;
      scalarValue = 0;
      traversed += stepLength;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  
	  Color tempcolr( AssociateValue( scalarValue, 1 ),
			 AssociateValue( scalarValue, 2 ),
			 AssociateValue( scalarValue, 3 ) );

	  colr = tempcolr;

	  opacity = AssociateValue( scalarValue, 0 );
	  
	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = geom->colorbuffer->get_pixel(px,py) * contribution +
      accumulatedColor;

  return accumulatedColor;
#endif  
}


/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
LevoyS::PerspectiveTrace( int from, int till )
{
  int loop, pool;
  Vector rayToTrace;
  Color pixelColor;
  Point beg, end;

  for ( pool = from; pool < till; pool++ )
    {
      for ( loop = 0; loop < yres; loop++ )
	{
	  // slightly alter the direction of ray
	  
	  rayToTrace = homeRay;
	  rayToTrace += rayIncrementU * ( pool - xres/2 );
	  rayToTrace += rayIncrementV * ( loop - yres/2 );

	  if ( box.Intersect( eye, rayToTrace, beg ) )
	    {
	      // TEMP!!!  perhaps i still want to check what the
	      // SV is at the point (it must be the corner)
	      // i will need to REMOVE this later

	      if ( beg.InInterval( end, 1.e-5 ) )
		pixelColor = geom->colorbuffer->get_pixel( pool, loop );
	      else
		pixelColor = (this->*CastRay)( eye, rayToTrace,
					      beg, end, pool, loop );
	    }
	  else
	    pixelColor = geom->colorbuffer->get_pixel( pool, loop );

	  // assign color value to pixel

	  (Image( loop, pool )).red =
	    (char)(pixelColor.r()*255);
	  (Image( loop, pool )).green =
	    (char)(pixelColor.g()*255);
	  (Image( loop, pool )).blue =
	    (char)(pixelColor.b()*255);

	}
    }
  
}



/**************************************************************************
 *
 * traces rays and builds the image.  myview contains the eye point,
 * look at point, the up vector, and the field of view.  x,y are the
 * raster sizes in the horizontal and vertical directions.  projection
 * type of 0 corresponds to an orthogonal projection, while 1
 * corresponds to a perspective projection.
 *
 **************************************************************************/

void
Levoy::TraceRays ( int projectionType )
{
  // cast one ray per pixel to determine pixel color

  if ( projectionType )
    {
      PerspectiveTrace( 0, xres );
    }
}


void
ContinuousRedraw()
{
}
