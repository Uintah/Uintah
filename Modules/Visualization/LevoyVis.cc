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
#include <Math/MiscMath.h>


/* the tiny epsilon vector added to counteract possible
   calculation error */

Vector EpsilonVector( 1.e-6, 1.e-6, 1.e-6 );

/* the most tiny opacity contribution of a point in
   the volume */

double EpsilonContribution = 0.01;


/**************************************************************************
 *
 * Constructor
 *
 **************************************************************************/

Levoy::Levoy ( ScalarFieldRG * grid, ColormapIPort * c,
	      Color& bg, Array1<double> * Xarr, Array1<double> * Yarr )
{
  int i;

  homeSFRGrid = grid;

  // set up the bounding box for the image

  homeSFRGrid->get_bounds( gmin, gmax );
  box.extend( gmin );
  box.extend( gmax );

  box.SetupSides();

  // get the colormap, if attached
  
  colormapFlag = c->get(cmap);

  backgroundColor = bg;

  ScalarVals     = Xarr;
  AssociatedVals = Yarr;

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
  
  Image = new Array2<CharColor>;
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
Levoy::DetermineRayStepSize ()
{
  double small[3];
  double result;
  
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
			       const int& rasterX, const int& rasterY )
{
  myview.get_normalized_viewplane( rayIncrementU, rayIncrementV );

  double aspect = double(rasterX) / double(rasterY);
  double fovy=RtoD(2*Atan(aspect*Tan(DtoR(myview.fov()/2.))));
  
  double lengthY = 2 * tan( DtoR( fovy / 2 ) );

  rayIncrementV *= lengthY / rasterY;
  rayIncrementU *= lengthY / rasterY;
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

  // compare distances

  if ( Abs( gmin.x() - e.x() ) > Abs( gmax.x() - e.x() ) )
    p.x( gmin.x() );
  else
    p.x( gmax.x() );
  
  if ( Abs( gmin.y() - e.y() ) > Abs( gmax.y() - e.y() ) )
    p.y( gmin.y() );
  else
    p.y( gmax.y() );
  
  if ( Abs( gmin.z() - e.z() ) > Abs( gmax.z() - e.z() ) )
    p.z( gmin.z() );
  else
    p.z( gmax.z() );
  
  return (p-e).length();
}



/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
Levoy::SetUp ( const View& myview, const int& x, const int& y )
{
  // temporary storage for eye position

  eye = myview.eyep();

  // the original ray from the eye point to the lookat point

  homeRay = myview.lookat() - eye;
  homeRay.normalize();

  cerr << "NORMALIZED is " << homeRay << endl;
  homeRay *= cos( DtoR( myview.fov() / 2 ) );

  cerr << "The home ray is: " << homeRay  <<endl;

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space

  rayStep = DetermineRayStepSize ();

  // the length of longest possible ray

  dmax = DetermineFarthestDistance ( eye );

  // create and initialize the array containing the image

  this->x = x;
  this->y = y;
  
  CharColor temp;

  // deallocate old array and allocate enough space

  Image->newsize( y, x );

  // initialize to the default (black) color

  Image->initialize( temp );

  // calculate the increments to be added in the u,v directions

  CalculateRayIncrements( myview, rayIncrementU, rayIncrementV, x, y );

  // determine the function to use

  if ( colormapFlag )
    CastRay = &Levoy::Ten;
  else if ( whiteFlag )
    CastRay = &Levoy::Eight;
  else // user defined SV-color map
    CastRay = &Levoy::Nine;
}


/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
Levoy::SetUp ( const View& myview, const int& x, const int& y,
	       Array2<double>* db, Array2<Color>*  cb, double Cnear,
	      double Cfar )
{
}


/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
Levoy::SetUp ( GeometryData * g )
{
}

/**************************************************************************
 *
 * Associates a value for the particular scalar value
 * given an array of scalar values and corresponding opacity/color values.
 *
 **************************************************************************/


double
Levoy::AssociateValue ( double scalarValue,
		       const Array1<double>& SVA, const Array1<double>& OA )
{
  double opacity;
  int i;
  
  // set to a bogus value in order to easily recognize
  // the case of scalarValue between the next-to-last and
  // last nodes.
  
  opacity = -1;

  for ( i = 1; i < SVA.size()-1; i++ )
    {
      if ( scalarValue < SVA[i] )
	{
	  opacity = ( OA[i] - OA[i-1] ) /
	    ( SVA[i] - SVA[i-1] )         *
	      ( scalarValue - SVA[i-1] )      +
		OA[i-1];
	  break;
	}
    }

  // if this is true, then the scalar value is between
  // next-to-last and last nodes
  
  if ( opacity == -1 )
    {
      i = SVA.size() - 1;
      opacity = ( OA[i] - OA[i-1] ) /
	( SVA[i] - SVA[i-1] )         *
	  ( scalarValue - SVA[i-1] )      +
	    OA[i-1];
    }

  return opacity;

}



/**************************************************************************
 *
 * calculates the pixel color by casting a ray from eye, going in the
 * direction step and proceeding at intervals of length rayStep.
 * the opacity map is not limited to 5 nodes.
 * the color associated with an opacity is retrieved from the
 * attached colormap module.
 *
 **************************************************************************/


Color
Levoy::Five ( const Point& eye, Vector& step,
	     const Point& beg, const Point& end )
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
  
  while ( contribution > EpsilonContribution &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{

	  // a MaterialHandle is retrieved through cmap->lookup(SV)
	  
	  colr = ( cmap->lookup( scalarValue ))->diffuse;

	  opacity = AssociateValue( scalarValue,
				   ScalarVals[0], AssociatedVals[0] );

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

  return accumulatedColor;
}





/**************************************************************************
 *
 * calculates the pixel color by casting a ray from eye, going in the
 * direction step and proceeding at intervals of length rayStep.
 * the opacity map is not limited to 5 nodes.
 * the color associated with any SV is white, while the opacity varies
 *
 **************************************************************************/




Color
Levoy::Six ( const Point& eye, Vector& step,
	     const Point& beg, const Point& end )
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

  Color colr = WHITE;

  double opacity;


  // keep going through the volume until the contribution of the voxel
  // color is almost none.  also, keep going through if the we are still
  // in the volume

  while ( contribution > EpsilonContribution &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  opacity = AssociateValue( scalarValue, ScalarVals[0],
				   AssociatedVals[0] );

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

  return accumulatedColor;
}




/**************************************************************************
 *
 * calculates the pixel color by casting a ray from eye, going in the
 * direction step and proceeding at intervals of length rayStep.
 * the opacity map is not limited to 5 nodes.
 * opacity and color vary according with the transfer map specified
 * by the user.
 *
 **************************************************************************/


Color
Levoy::Seven ( const Point& eye, Vector& step,
	     const Point& beg, const Point& end )
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
  
  while ( contribution > EpsilonContribution &&  ( atPoint - eye ).length() < dmax )
    {
      atPoint += step;
      scalarValue = 0;

      if ( homeSFRGrid->interpolate( atPoint, scalarValue ) )
	{
	  
	  Color tempcolr( AssociateValue( scalarValue,
					 ScalarVals[1], AssociatedVals[1] ),
			 AssociateValue( scalarValue,
					ScalarVals[2], AssociatedVals[2] ),
			 AssociateValue( scalarValue,
					ScalarVals[3], AssociatedVals[3] ) );

	  colr = tempcolr;

	  opacity = AssociateValue( scalarValue,ScalarVals[0],
				   AssociatedVals[0] );
	  
	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

  return accumulatedColor;
}





/**************************************************************************
 *
 * calculates the pixel color by casting a ray starting at the first
 * intersection of the ray with the bbox, and ending at the second
 * intersection of the ray with the bbox.
 *
 * the opacity map is not limited to 5 nodes.
 * opacity varies according with the transfer map specified
 * by the user.  the color associated with any scalar value
 * is fixed to be white.
 *
 **************************************************************************/

Color
Levoy::Eight ( const Point& eye, Vector& step,
	     const Point& beg, const Point& end )
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

  double mylength;

  // the length of ray stepped through so far

  double traversed;

  // contribution of the first voxel's color and opacity
  
  contribution = 1.0;
  
  // since no voxels have been crossed, no color has been
  // accumulated.

  accumulatedColor = BLACK;

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

  // yet another check for whether i calculated the dmax correctly...
  // actually, i probably didn't calculate it correctly...

  if ( mylength + ( eye - beg ).length() + 1.e-5 > dmax )
    {
      cerr << "DMAX not calculated correctly: " << dmax << " vs " << mylength + ( eye - beg ).length() + 1.e-5 << endl;
      cerr << beg <<"   " << end << endl;
    }

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
	  opacity = AssociateValue( scalarValue, ScalarVals[0],
				   AssociatedVals[0] );

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel.  if contribution is minute, the background
  // is not visible therefore don't add the background color
  // to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

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
Levoy::Nine ( const Point& eye, Vector& step,
	     const Point& beg, const Point& end )
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

  // step length

  double stepLength;

  // the length of the ray to be cast

  double mylength;

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
	  
	  Color tempcolr( AssociateValue( scalarValue,
					 ScalarVals[1], AssociatedVals[1] ),
			 AssociateValue( scalarValue,
					ScalarVals[2], AssociatedVals[2] ),
			 AssociateValue( scalarValue,
					ScalarVals[3], AssociatedVals[3] ) );

	  colr = tempcolr;

	  opacity = AssociateValue( scalarValue, ScalarVals[0],
				   AssociatedVals[0] );
	  
	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

  return accumulatedColor;
}




/**************************************************************************
 *
 * calculates the pixel color by casting a ray starting at the first
 * intersection of the ray with the bbox, and ending at the second
 * intersection of the ray with the bbox.
 *
 * the color associated with an opacity is retrieved from the
 * attached colormap module.
 *
 **************************************************************************/


Color
Levoy::Ten ( const Point& eye, Vector& step,
	     const Point& beg, const Point& end )
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

  // step length

  double stepLength;

  // the length of the ray to be cast

  double mylength;

  // the length of ray stepped through so far

  double traversed;

  // the color corresponding to the scalar value

  Color colr;

  // the opacity corresponding to the scalar value

  double opacity;

  
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

  // find the step length

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

	  // a MaterialHandle is retrieved through cmap->lookup(SV)
	  
	  colr = ( cmap->lookup( scalarValue ))->diffuse;

	  opacity = AssociateValue( scalarValue,
				   ScalarVals[0], AssociatedVals[0] );

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

  return accumulatedColor;
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
  Point beg, end;

  cerr <<"CALLED a fnc in LEVOY\n";
  
  for ( pool = from; pool < till; pool++ )
    {
      for ( loop = 0; loop < y; loop++ )
	{
	  // slightly alter the direction of ray
	  
	  rayToTrace = homeRay;
	  rayToTrace += rayIncrementU * ( pool - x/2 );
	  rayToTrace += rayIncrementV * ( loop - y/2 );

	  if ( box.Intersect( eye, rayToTrace, beg, end ) )
	    {
	      // TEMP!!!  perhaps i still want to check what the
	      // SV is at the point (it must be the corner)
	      // i will need to REMOVE this later

	      if ( ( eye - beg ).length() > ( eye - end ).length() )
		{
		  cerr << "swapping!!!!! ...\n";
		  Point aaa;
		  aaa = beg;
		  beg = end;
		  end = aaa;
		}

	      if ( beg.InInterval( end, 1.e-5 ) )
		pixelColor = backgroundColor;
	      else
		pixelColor = (this->*CastRay)( eye, rayToTrace,
					      beg, end );
	    }
	  else
	    pixelColor = backgroundColor;

	  /*
	     i need to do it this way, unless i can write
	     an operator overload fnc for ()=
	     Image( loop, pool ) returns a value; it cannot
	     be assigned.
	     Image( loop, pool ) = pixelColor;
	     */

	  ((*Image)( loop, pool )).red =
	    (char)(pixelColor.r()*255);
	  ((*Image)( loop, pool )).green =
	    (char)(pixelColor.g()*255);
	  ((*Image)( loop, pool )).blue =
	    (char)(pixelColor.b()*255);

	}
    }
  
}



/**************************************************************************
 *
 * Constructor
 *
 **************************************************************************/

LevoyS::LevoyS ( ScalarFieldRG * grid, ColormapIPort * c,
	      Color& bg, Array1<double> * Xarr, Array1<double> * Yarr )
: Levoy( grid, c, bg, Xarr, Yarr )
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
LevoyS::SetUp ( const View& myview, const int& x, const int& y,
	       Array2<double>* db, Array2<Color>*  cb, double Cnear,
	      double Cfar )
{
  // temporary storage for eye position

  eye = myview.eyep();

  // the original ray from the eye point to the lookat point

  homeRay = myview.lookat() - eye;
  homeRay.normalize();

  cerr << "NORMALIZED is " << homeRay << endl;
  homeRay *= cos( DtoR( myview.fov() / 2 ) );

  cerr << "The home ray is: " << homeRay  <<endl;
  

  // the distance between the 2 clipping planes
  
  clipConst = ( Cfar - Cnear ) / homeRay.length();

  // the ratio of near clipping plane to homeRaylength

  nearTohome = Cnear / homeRay.length();
  

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space

  rayStep = DetermineRayStepSize ();

  // the length of longest possible ray

  dmax = DetermineFarthestDistance ( eye );

  // create and initialize the array containing the image

  this->x = x;
  this->y = y;
  
  CharColor temp;

  // keep the depth and color buffer information

  this->DepthBuffer = db;
  this->BackgroundBuffer = cb;

  // deallocate, allocate, and initialize the array
  
  Image->newsize( y, x );
  
  Image->initialize( temp );

  // calculate the increments to be added in the u,v directions

  CalculateRayIncrements( myview, rayIncrementU, rayIncrementV, x, y );

  // determine the function to use

  if ( colormapFlag )
    CastRay = &LevoyS::Thirteen;
  else if ( whiteFlag )
    CastRay = &LevoyS::Eleven;
  else // user defined SV-color map
    CastRay = &LevoyS::Twelve;
}


/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
LevoyS::SetUp ( GeometryData * g )
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

  cerr << "Far clipping plane is " << g->zfar << endl;
  cerr << "Near clipping plane is " << g->znear << endl;
  cerr << "clipConst ended up being: " << clipConst << endl;

  // the ratio of near clipping plane to homeRaylength

  nearTohome = g->znear / homeRay.length();

  cerr << "near to home constant is " << nearTohome << endl;

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space

  rayStep = DetermineRayStepSize ();

  // the length of longest possible ray

  dmax = DetermineFarthestDistance ( eye );

  // create and initialize the array containing the image

  this->x = g->xres;
  this->y = g->yres;
  
  CharColor temp;

  // keep the depth and color buffer information

/*  
  this->DepthBuffer = db;
  this->BackgroundBuffer = cb;
  */

  // deallocate, allocate, and initialize the array
  
  Image->newsize( g->yres, g->xres );
  
  Image->initialize( temp );

  // calculate the increments to be added in the u,v directions

//  CalculateRayIncrements( myview, rayIncrementU, rayIncrementV, x, y );
  CalculateRayIncrements( *(g->view), rayIncrementU, rayIncrementV,
			 g->xres, g->yres );

  // determine the function to use

  if ( colormapFlag )
    CastRay = &LevoyS::Thirteen;
  else if ( whiteFlag )
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

  double mylength;

  // the length of ray stepped through so far

  double traversed;

  // contribution of the first voxel's color and opacity
  
  contribution = 1.0;
  
  // since no voxels have been crossed, no color has been
  // accumulated.

  accumulatedColor = BLACK;

  
  
  /* DAVES */

  // calculate the length from the eye to the DepthBuffer based point
  // in space

/*   int templength = step.length() * ( clipConst * DepthBuffer(x,y) +
     nearTohome ); */

  int templength = step.length() * ( clipConst *
		   geom->depthbuffer->get_depth(px,py) + nearTohome );

  // compare the 2 lengths, pick the shorter one

  if ( templength < mylength )
    {
      mylength = templength;
      cerr << "U";
    }
    
  /* END DAVES */


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

  // yet another check for whether i calculated the dmax correctly...
  // actually, i probably didn't calculate it correctly...

  if ( mylength + ( eye - beg ).length() + 1.e-5 > dmax )
    {
      cerr << "DMAX not calculated correctly: " << dmax << " vs " << mylength + ( eye - beg ).length() + 1.e-5 << endl;
      cerr << beg <<"   " << end << endl;
    }

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
	  opacity = AssociateValue( scalarValue, ScalarVals[0],
				   AssociatedVals[0] );

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel.  if contribution is minute, the background
  // is not visible therefore don't add the background color
  // to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

    // DAVES
    //    accumulatedColor = BackgroundBuffer(px,py) * contribution + accumulatedColor;

  /* DAVES */

  accumulatedColor = geom->colorbuffer->get_pixel(px,py) * contribution +
    accumulatedColor;

  /* END DAVES */

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

  double mylength;

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

  /* DAVES

     // calculate the length from the eye to the DepthBuffer based point
     // in space

     int templength = step.length() * ( clipConst * DepthBuffer(x,y) +
     nearTohome );

     // compare the 2 lengths, pick the shorter one

  if ( templength < mylength )
    mylength = templength;
  else
    cerr << "T";
    
    */

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
	  
	  Color tempcolr( AssociateValue( scalarValue,
					 ScalarVals[1], AssociatedVals[1] ),
			 AssociateValue( scalarValue,
					ScalarVals[2], AssociatedVals[2] ),
			 AssociateValue( scalarValue,
					ScalarVals[3], AssociatedVals[3] ) );

	  colr = tempcolr;

	  opacity = AssociateValue( scalarValue, ScalarVals[0],
				   AssociatedVals[0] );
	  
	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

  // DAVES  
//    accumulatedColor = BackgroundBuffer(px,py) * contribution + accumulatedColor;

  return accumulatedColor;
}





/**************************************************************************
 *
 * calculates the pixel color by casting a ray starting at the first
 * intersection of the ray with the bbox, and ending at the second
 * intersection of the ray with the bbox.
 *
 * the color associated with an opacity is retrieved from the
 * attached colormap module.
 *
 **************************************************************************/


Color
LevoyS::Thirteen ( const Point& eye, Vector& step,
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

  // step length

  double stepLength;

  // the length of the ray to be cast

  double mylength;

  // the length of ray stepped through so far

  double traversed;

  // the color corresponding to the scalar value

  Color colr;

  // the opacity corresponding to the scalar value

  double opacity;

  
  // contribution of the first voxel's color and opacity
  
  contribution = 1.0;
  
  // begin trace at the eye (somewhat inefficient, TEMP)
  
  atPoint = eye;

  // since no voxels have been crossed, no color has been
  // accumulated.

  accumulatedColor = BLACK;

  /* DAVES

     // calculate the length from the eye to the DepthBuffer based point
     // in space

     int templength = step.length() * ( clipConst * DepthBuffer(x,y) +
     nearTohome );

     // compare the 2 lengths, pick the shorter one

  if ( templength < mylength )
    mylength = templength;
  else
    cerr << "T";
    
    */

  // find step vector
  
  step.normalize();

  step *= rayStep;

  // find the step length

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

	  // a MaterialHandle is retrieved through cmap->lookup(SV)
	  
	  colr = ( cmap->lookup( scalarValue ))->diffuse;

	  opacity = AssociateValue( scalarValue,
				   ScalarVals[0], AssociatedVals[0] );

	  accumulatedColor += colr * ( contribution * opacity );
	  contribution = contribution * ( 1.0 - opacity );
	}
    }

  // background color should contribute to the final color of
  // the pixel
  // if contribution is minute, the background is not visible
  // therefore don't add the background color to the accumulated one.
  
  if ( contribution > EpsilonContribution )
    accumulatedColor = backgroundColor * contribution + accumulatedColor;

// DAVES    
//    accumulatedColor = BackgroundBuffer(px,py) * contribution + accumulatedColor;

  return accumulatedColor;
}




/**************************************************************************
 *
 *
 *
 **************************************************************************/

void
Levoy::ParallelTrace( int from, int till )
{
  Vector normal;
  
  normal = Cross( rayIncrementV, rayIncrementU );
  normal.normalize();

  // TEMP! THIS IS DONE INCORRECTLY!!!

  cerr << "The orthogonal projection does not work correctly\n";
  
  return;
  
  int loop, pool;
  Point startAt;
  Color pixelColor;
  Point beg, end;

  for ( pool = from; pool < till; pool++ )
    {
      for ( loop = 0; loop < y; loop++ )
	{
	  // slightly alter the eye point/ pixel position
	  
	  startAt = eye;
	  startAt += rayIncrementU * ( pool - x/2 );
	  startAt += rayIncrementV * ( loop - y/2 );

	  if ( box.Intersect( startAt, homeRay, beg, end ) )
	    {
	      // TEMP!!!  perhaps i still want to check what the
	      // SV is at the point (it must be the corner)
	      
	      if ( beg.InInterval( end, 1.e-5 ) )
		pixelColor = backgroundColor;
	      else
		pixelColor = (this->*CastRay)( startAt, homeRay,
					      beg, end );
	    }
	  else
	    pixelColor = backgroundColor;

	  /*
	     i need to do it this way, unless i can write
	     an operator overload fnc for ()=
	     Image( loop, pool ) returns a value; it cannot
	     be assigned.
	     Image( loop, pool ) = pixelColor;
	     */

	  ((*Image)( loop, pool )).red =
	    (char)(pixelColor.r()*255);
	  ((*Image)( loop, pool )).green =
	    (char)(pixelColor.g()*255);
	  ((*Image)( loop, pool )).blue =
	    (char)(pixelColor.b()*255);

	}
    }
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

  cerr <<"CALLED a fnc in levoyS\n";
  
  for ( pool = from; pool < till; pool++ )
    {
      for ( loop = 0; loop < y; loop++ )
	{
	  // slightly alter the direction of ray
	  
	  rayToTrace = homeRay;
	  rayToTrace += rayIncrementU * ( pool - x/2 );
	  rayToTrace += rayIncrementV * ( loop - y/2 );

	  if ( box.Intersect( eye, rayToTrace, beg, end ) )
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

	  ((*Image)( loop, pool )).red =
	    (char)(pixelColor.r()*255);
	  ((*Image)( loop, pool )).green =
	    (char)(pixelColor.g()*255);
	  ((*Image)( loop, pool )).blue =
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

Array2<CharColor> *
Levoy::TraceRays ( int projectionType )
{
  // cast one ray per pixel to determine pixel color

  if ( projectionType )
    {
      PerspectiveTrace( 0, x );
    }
  else
    {
      ParallelTrace( 0, x );
    }

  return Image;
}
