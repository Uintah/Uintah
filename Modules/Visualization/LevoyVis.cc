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
 */

#include "kuswik.h"
#include <Modules/Visualization/LevoyVis.h>






/**************************************************************************
 *
 * constructor
 *
**************************************************************************/

Levoy::Levoy ( ScalarFieldRG * grid, ColormapIPort * c,
	      Color& bg, Array1<double> Xarr, Array1<double> Yarr )
{
  homeSFRGrid = grid;

  colormapFlag = c->get(cmap);

  backgroundColor = bg;

  SVArray = Xarr;
  Opacity = Yarr;
}



/**************************************************************************
 *
 * attempts to determine the most appropriate interval which is used
 * to step through the volume.
 *
**************************************************************************/

double
Levoy::DetermineRayStepSize ()
{
  Point gmin, gmax;
  double small[3];
  double result;
  
  homeSFRGrid->get_bounds( gmin, gmax );
  
  // calculate a step size that is about the length of one voxel
  
  small[0] = ( gmax.x() - gmin.x() ) / ( homeSFRGrid->nx  );
  small[1] = ( gmax.y() - gmin.y() ) / ( homeSFRGrid->ny  );
  small[2] = ( gmax.z() - gmin.z() ) / ( homeSFRGrid->nz  );

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

  return result;
}





/**************************************************************************
 *
 * part of a ray tracer; calculates the ray increments in the u,v directions
 *
**************************************************************************/

void
Levoy::CalculateRayIncrements ( View myview,
			Vector& rayIncrementU, Vector& rayIncrementV,
			       int rasterX, int rasterY )
{

  double length;

  // assuming that fov represents half the angle
  // calculate the length of view plane given an angle
  
  length = 2 * sin( DtoR( myview.fov() ) );

  myview.get_normalized_viewplane( rayIncrementU, rayIncrementV );

  rayIncrementU = rayIncrementU * length/rasterX;
  rayIncrementV = rayIncrementV * length/rasterY;
}





/**************************************************************************
 *
 * determine the longest ray possible.  necessary for loop termination
 * during calculation of pixel color.
 *
**************************************************************************/

double
Levoy::DetermineFarthestDistance ( const Point& e )
{
  Point p;
  Point gmin, gmax;

  // get the bounding box
  
  homeSFRGrid->get_bounds( gmin, gmax );

  if ( abs( gmin.x() - e.x() ) > abs( gmax.x() - e.x() ) )
    p.x( gmin.x() );
  else
    p.x( gmax.x() );
  
  if ( abs( gmin.y() - e.y() ) > abs( gmax.y() - e.y() ) )
    p.y( gmin.y() );
  else
    p.y( gmax.y() );
  
  if ( abs( gmin.z() - e.z() ) > abs( gmax.z() - e.z() ) )
    p.z( gmin.z() );
  else
    p.z( gmax.z() );

  return (p-e).length();
}



/**************************************************************************
 *
 * Uses Han-Wei's method which i used to check if my implementation
 * works.  Does not take into account varying background color.
 * (the background color is always black)
 *
**************************************************************************/

Color&
Levoy::Four( Point eye, Vector ray, double rayStep )
{

  
  // this point represents the point in the volume
  Point atPoint;
  
  double scalarValue;
  double contribution;
  double epsilon;

  // this is the tiny step vector added to the atPoint each time
  Vector step( ray );

  // epsilon should be defined globally, outside of this fnc
  //????
  epsilon = 0.01;
  
  contribution = 1.0;

  // THINK a bit more about different background Colors...
  // background color should be defined globally as well
  
  // i don't know what the range of color in sci run is yet
  //  Color backgroundColor( 0, 0, 0 );

  // initially, assign accumulated color to be black
  Color accumulatedColor( 0., 0., 0. );

  // find step vector
  
  step.normalize();

  step = step * rayStep;

  // initialize atPoint to eye
  
  atPoint = eye;


  MaterialHandle col;

  Color temp( 1, 1, 1 );
  Color colr;

  double opacity, alpha, accAlpha;
  alpha = 0.;
  accAlpha = 0.;


  // keep going through the volume until the contribution of the voxel
  // color is almost none.  also, keep going through if the we are still
  // in the volume

  while ( contribution > epsilon &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  opacity = 0;
	  
	  if ( scalarValue  < SVArray[1]    )
	    opacity = Opacity[0] +
	      ( Opacity[1] - Opacity[0] ) / ( SVArray[1] - SVArray[0] ) *
		( scalarValue - SVArray[0] );

	  if ( scalarValue >= SVArray[1] &&
	      scalarValue < SVArray[2]      )
	    opacity = Opacity[1] +
	      ( Opacity[2] - Opacity[1] ) / ( SVArray[2] - SVArray[1] ) *
		( scalarValue - SVArray[1] );
	  
	  if ( scalarValue >= SVArray[2]   )
	    opacity = Opacity[2] +
	      ( Opacity[3] - Opacity[2] ) / ( SVArray[3] - SVArray[2] ) *
		( scalarValue - SVArray[2] );

	  if ( scalarValue >= SVArray[3]   )
	    opacity = Opacity[3] +
	      ( Opacity[4] - Opacity[3] ) / ( SVArray[4] - SVArray[3] ) *
		( scalarValue - SVArray[3] );

	  colr = temp;
	  
	  if ( opacity != 0 )
	    {
	      alpha = opacity * ( 1 - accAlpha );
	      accumulatedColor += colr * alpha;
	      accAlpha += alpha;
	    }
	}
    }
  
  return accumulatedColor;

}






/**************************************************************************
 *
 * calculates the pixel color by casting a ray from eye, going in the
 * direction step and proceeding at intervals of length rayStep.
 * the opacity map is not limited to 5 nodes.
 *
**************************************************************************/


Color&
Levoy::Five ( Point eye, Vector step, double rayStep )
{

  int i;

  // if the voxel's contribution value is below epsilon,
  // it's color and opacity have no impact on the color
  // of pixel
  
  double epsilon;

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

  
  // epsilon should be defined globally, outside of this fnc
  //TEMP

  epsilon = 0.01;

  // contribution of the first voxel's color and opacity
  
  contribution = 1.0;
  
  // begin trace at the eye (somewhat inefficient, TEMP)
  
  atPoint = eye;

  // since no voxels have been crossed, no color has been
  // accumulated.

  accumulatedColor = BLACK;

  // find step vector
  
  step.normalize();

  step *= rayStep;

  Color colr;

  double opacity;

  // keep going through the volume until the contribution of the voxel
  // color is almost none.  also, keep going through if the we are still
  // in the volume
  
  while ( contribution > epsilon &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{

	  // a MaterialHandle is retrieved through cmap->lookup(SV)
	  // but only if the colormap is connected.  otherwise,
	  // use white
	  
	  if ( colormapFlag )
	      colr = ( cmap->lookup( scalarValue ))->diffuse;
	  else
	      colr = WHITE;

	  // set to a bogus value in order to easily recognize
	  // the case of scalarValue between the next-to-last and
	  // last nodes.
	  
	  opacity = -1;

	  for ( i = 1; i < SVArray.size()-1; i++ )
	    {
	      if ( scalarValue < SVArray[i] )
		{
		  opacity = ( Opacity[i] - Opacity[i-1] ) /
		    ( SVArray[i] - SVArray[i-1] )         *
		      ( scalarValue - SVArray[i-1] )      +
		      Opacity[i-1];
		  break;
		}
	    }

	  // if this is true, then the scalar value is between
	  // next-to-last and last nodes
	  
	  if ( opacity == -1 )
	    {
	      i = SVArray.size() - 1;
	      opacity = ( Opacity[i] - Opacity[i-1] ) /
		( SVArray[i] - SVArray[i-1] )         *
		  ( scalarValue - SVArray[i-1] )      +
		    Opacity[i-1];
	    }

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
       }
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > epsilon )
      accumulatedColor = backgroundColor * contribution + accumulatedColor;

  return accumulatedColor;
}







/**************************************************************************
 *
 * casts a ray in order to determine pixel color.
 *
**************************************************************************/

Color&
Levoy::CastRay ( Point eye, Vector ray, double rayStep )
{
  Color hmm, imm;
  
    hmm = Five( eye, ray, rayStep );
  
  imm = Four( eye, ray, rayStep );

  if ( backgroundColor != BLACK || colormapFlag )
    return( hmm );

  if ( ! hmm.InInterval( imm, 0.001 ) )
    {
      cout << "Hmm and Imm are different\n";
      cout << "Five: " << hmm.r() << " " << hmm.g() << " " << hmm.b() << endl;
      cout << "Four: " << imm.r() << " " << imm.g() << " " << imm.b() << endl;
    }
  
  return(hmm);
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

Array2<CharColor> *
Levoy::TraceRays ( View myview, int x, int y, int projectionType )
{
  
  int loop, pool;
  Vector rayToTrace;
  Color pixelColor;

  // the ray increment that is added to the original vector from
  // the eye to the lookat point.  used for perspective projection.
  
  Vector rayIncrementU, rayIncrementV;

  // temporary storage for eye position

  Point eye( myview.eyep() );

  Point startAt( eye );

  // the original ray from the eye point to the lookat point

  Vector homeRay = myview.lookat() - eye;
  homeRay.normalize();

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space
  
  double rayStep = DetermineRayStepSize ();

  // the length of longest possible ray

  dmax = DetermineFarthestDistance ( eye );
  
  // create and initialize the array containing the image
  
  CharColor temp;
  Array2<CharColor> * Image = new Array2<CharColor>;
  Image->newsize( y, x );
  Image->initialize( temp );

  // calculate the increments to be added in the u,v directions

  CalculateRayIncrements( myview, rayIncrementU, rayIncrementV, x, y );

  // cast one ray per pixel to determine pixel color

  if ( projectionType )
  {

    for ( loop = 0; loop < y; loop++ )
      {
	for ( pool = 0; pool < x; pool++ )
	  {

	    // slightly alter the direction of ray
	    
	    rayToTrace = homeRay;
	    rayToTrace += rayIncrementU * ( pool - x/2 );
	    rayToTrace += rayIncrementV * ( loop - y/2 );

	    	cerr << ".";

	    pixelColor = CastRay( eye, rayToTrace, rayStep );

	    /*
	       i need to do it this way, unless i can write
	       an operator overload fnc for ()=
	       Image( loop, pool ) returns a value; it cannot
	       be assigned.
	       Image( loop, pool ) = pixelColor;
	       */

	    ((*Image)( loop, pool )).red = (char)(pixelColor.r()*255);
	    ((*Image)( loop, pool )).green = (char)(pixelColor.g()*255);
	    ((*Image)( loop, pool )).blue = (char)(pixelColor.b()*255);

	  }
	    cerr << loop << endl;
      }
      cerr << endl;

  }
  else
  {

    for ( loop = 0; loop < y; loop++ )
      {
	for ( pool = 0; pool < x; pool++ )
	  {
	    // slightly alter the eye point/ pixel position
	    
	    startAt = eye;
	    startAt += rayIncrementU * ( pool - x/2 );
	    startAt += rayIncrementV * ( loop - y/2 );

	    pixelColor = CastRay( startAt, homeRay, rayStep );

	    /*
	       i need to do it this way, unless i can write
	       an operator overload fnc for ()=
	       Image( loop, pool ) returns a value; it cannot
	       be assigned.
	       Image( loop, pool ) = pixelColor;
	       */

	    ((*Image)( loop, pool )).red = (char)(pixelColor.r()*255);
	    ((*Image)( loop, pool )).green = (char)(pixelColor.g()*255);
	    ((*Image)( loop, pool )).blue = (char)(pixelColor.b()*255);

	  }
      }
  }

  return Image;
}
