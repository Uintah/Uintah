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

Levoy::Levoy ( ScalarFieldRG * grid )
{
  homeSFRGrid = grid;
}

void
Levoy::AssignRaster ( int x, int y )
{
  rasterX = x;
  rasterY = y;
}

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


void
Levoy::CalculateRayIncrements ( View myview,
			Vector& rayIncrementU, Vector& rayIncrementV )
{

  double length;

  // assuming that fov represents half the angle
  // calculate the length of view plane given an angle
  
  length = 2* sqrt( 1 - cos( DtoR( myview.fov() ) )
		      * cos( DtoR( myview.fov() ) ) );

  myview.get_normalized_viewplane( rayIncrementU, rayIncrementV );

  rayIncrementU = rayIncrementU * length/rasterX;
  rayIncrementV = rayIncrementV * length/rasterY;

//  cout << "Ray increment in U is: " << rayIncrementU << endl;
//  cout << "Ray increment in V is: " << rayIncrementV << endl;
}


  
void
Levoy::TraceRays ( View myview,
	   Array2<char>& image, Array2<CharColor>& Image,
	   Color& bgColor, double Xval[], double Yval[] )
{
  int loop, pool;
  Vector rayToTrace;
  Color pixelColor;
  Vector rayIncrementU, rayIncrementV;

  // assign variables
  
  Point eye( myview.eyep() );

  CalculateRayIncrements( myview, rayIncrementU,
			 rayIncrementV );

  double rayStep = DetermineRayStepSize ();

  double farthest = DetermineFarthestDistance ( eye );

  // homeRay = the ray that points directly from eye to lookat pt
  
  Vector homeRay = myview.lookat() - myview.eyep();
  homeRay.normalize();

  at();
  cout << "dim1 : " << Image.dim1();
  cout << "\n dim2 : " << Image.dim2() << endl;
  cout << "rasterX: " << rasterX << endl;
  cout << "rasterY: " << rasterY << endl;
  
  for ( loop = 0; loop < rasterY; loop++ )
    {
    for ( pool = 0; pool < rasterX; pool++ )
      {
	rayToTrace = homeRay;
	rayToTrace += rayIncrementU * ( pool - rasterX/2 );
	rayToTrace += rayIncrementV * ( loop - rasterY/2 );

	pixelColor = CastRay( myview.eyep(), rayToTrace, rayStep, farthest,
			     bgColor, Xval, Yval );

	image( loop, pool ) = (char)(pixelColor.g()*255);
	
/* for some reason, this didn't work...
   newly added (actually, put into a comment...
   */
	Image( loop, pool ) = pixelColor;

	(Image( loop, pool )).red = (char)(pixelColor.r()*255);
	(Image( loop, pool )).green = (char)(pixelColor.g()*255);
	(Image( loop, pool )).blue = (char)(pixelColor.b()*255);

      }
  }
  
//  cout << endl;
}


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

//  cout << "farthest from eye " << p << endl;
  
  return (p-e).length();
}


Color&
Levoy::SecondTry ( Point eye, Vector ray, double rayStep, double dmax,
	   Color& backgroundColor )
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
  Color accumulatedColor( 0, 0, 0 );

  // find step vector
  
  step.normalize();

  step = step * rayStep;

  // initialize atPoint to eye
  
  atPoint = eye;

  // keep going through the volume until the contribution of the voxel
  // color is almost none.  also, keep going through if the we are still
  // in the volume

  MaterialHandle col;

  Color temp;
  
  while ( contribution > epsilon &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;
      
      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  if ( colormapFlag )
	    col = cmap->lookup( scalarValue );
	  else
	    col->diffuse = temp;
	  
//	  cout << "scalar value is: " << scalarValue << " ; ";

	  // note that col->diffuse and col->transparency correspond
	  // to the color and opacity of object
	  
	  accumulatedColor +=  col->diffuse * ( contribution * (1.0 - col->transparency) );
	  contribution -= contribution * ( 1.0 - (1.0 - col->transparency) );
       }
      else
	{
//	  cout << ".";
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > epsilon )
    {
//      cout << "contrib is " << contribution << endl;
      
      accumulatedColor = backgroundColor * contribution + accumulatedColor;
    }

//  cout << endl;

  return accumulatedColor;
}


Color&
Levoy::Three ( Point eye, Vector ray, double rayStep, double dmax,
	   Color& backgroundColor, double Xval[], double Yval[] )
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

  // keep going through the volume until the contribution of the voxel
  // color is almost none.  also, keep going through if the we are still
  // in the volume

  ////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////
  //

  int j;
  double Opacity[5];
  double SVArray[5];

  for ( j=0; j < 5; j++ )
    {
      Opacity[j] = ( 202. - Yval[j] ) / 202.;
      SVArray[j] = Xval[j] / 202. * ( maxSV - minSV ) + minSV;

//      cout << "opacity of: " << Opacity[j] << " SVArray of " << SVArray[j] << endl;
    }

  
  
  ////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////

  MaterialHandle col;

  Color temp( 1, 1, 1 );
  Color colr;

#if 0  
  double opacity, alpha, accAlpha;
  alpha = 0.;
  accAlpha = 0.;


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
#endif
  
#if 0  
  while ( contribution > epsilon &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  if ( scalarValue > 30 && scalarValue < 50 )
	    opacity = 0.5;
	  else
	    opacity = 0.;
	  
	  colr = temp;
	  
	  if ( opacity != 0 )
	    {
	      alpha = opacity * ( 1 - accAlpha );
	      accumulatedColor += colr * alpha;
	      accAlpha += alpha;
	    }
	}
    }
#endif      

  double opacity;
  
  while ( contribution > epsilon &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  if ( colormapFlag )
	    {
	      col = cmap->lookup( scalarValue );
	      colr = col->diffuse;
	    }
	  else
	    {
	      colr = temp;
	    }

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

	  // note that col->diffuse and col->transparency correspond
	  // to the color and opacity of object

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
       }
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > epsilon )
    {
//      cout << "contrib is " << contribution << endl;

//      cout << "%";
      
      accumulatedColor = backgroundColor * contribution + accumulatedColor;
    }

//  cout << endl;

  return accumulatedColor;
}


Color&
Levoy::CastRay ( Point eye, Vector ray, double rayStep, double farthest, Color& backgroundColor, double Xval[], double Yval[] )
{

  // newly added
  return ( Three( eye, ray, rayStep, farthest, backgroundColor, Xval, Yval ));

}
