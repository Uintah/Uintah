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

#include "kuswik.h"
#include <Classlib/Timer.h>
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
  int i, j;

  homeSFRGrid = grid;

  // set up the bounding box for the image

  homeSFRGrid->get_bounds( gmin, gmax );
  box.extend( gmin );
  box.extend( gmax );

  box.SetupSides();

  // get the colormap, if attached

  if ( c != NULL )
    colormapFlag = c->get(cmap);

  backgroundColor = bg;

  ScalarVals     = Xarr;
  AssociatedVals = Yarr;

  // do some processing on the array because AssociateValue
  // function takes a lot of time

  Slopes = new Array1<double> [4];

  // add a dummy at position 0

  Slopes[0].add( 0. );
  Slopes[1].add( 0. );
  Slopes[2].add( 0. );
  Slopes[3].add( 0. );

  for ( i = 0; i < 4; i++ )
    for ( j = 1; j < ScalarVals[i].size(); j++ )
      Slopes[i].add( ( AssociatedVals[i][j] - AssociatedVals[i][j-1] ) /
		    ( ScalarVals[i][j] - ScalarVals[i][j-1] ) );

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

  // setup the lighting coefficients

  ambient_coeff = 0.3;
  diffuse_coeff = 0.4;
  specular_coeff = 0.2;
  specular_iter = 30;
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
Levoy::SetUp ( const View& myview, int x, int y )
{
  // temporary storage for eye position

  eye = myview.eyep();

  // the original ray from the eye point to the lookat point

  homeRay = myview.lookat() - eye;
  homeRay.normalize();

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space

  rayStep = DetermineRayStepSize ();

  // the length of longest possible ray

  dmax = DetermineFarthestDistance ( eye );

  // make sure that the raster sizes do not exceed the OpenGL window
  // size

  if ( x > VIEW_PORT_SIZE )
    x = VIEW_PORT_SIZE;
  
  if ( y > VIEW_PORT_SIZE )
    y = VIEW_PORT_SIZE;

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

  // calculate the vector, L, which points toward the sun ( a point
  // light source which is very far away

//  Lvector = homeRay + rayIncrementU * 3 + rayIncrementV * 3;
  Lvector = homeRay;
  Lvector.normalize();
  
  // TEMP!!!!  2 different logic statements, interchanged,
  // in order to compare speed
  
  // determine the function to use

  if ( colormapFlag )
    CastRay = &Levoy::Ten;
  else if ( whiteFlag )
    CastRay = &Levoy::Eight;
  else // user defined SV-color map
    CastRay = &Levoy::Nine;
  
  if ( colormapFlag )
    CastRay = &Levoy::Ten;
  else if ( whiteFlag )
    CastRay = &Levoy::Bresenham2;
  else // user defined SV-color map
    CastRay = &Levoy::Nine;
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
				   ScalarVals[0], AssociatedVals[0],
				   Slopes[0] );

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
				   AssociatedVals[0], Slopes[0] );

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
					 ScalarVals[1], AssociatedVals[1],
					 Slopes[1] ),
			 AssociateValue( scalarValue,
					ScalarVals[2], AssociatedVals[2],
					Slopes[2] ),
			 AssociateValue( scalarValue,
					ScalarVals[3], AssociatedVals[3],
					Slopes[3] ),
			 );

	  colr = tempcolr;

	  opacity = AssociateValue( scalarValue,ScalarVals[0],
				   AssociatedVals[0], Slopes[0]  );
	  
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
 * is fixed to be WHITE.
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
				   AssociatedVals[0], Slopes[0] );

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
					 ScalarVals[1], AssociatedVals[1],
					 Slopes[1] ),
			 AssociateValue( scalarValue,
					ScalarVals[2], AssociatedVals[2],
					Slopes[2] ),
			 AssociateValue( scalarValue,
					ScalarVals[3], AssociatedVals[3],
					Slopes[3] ) );

	  colr = tempcolr;

	  opacity = AssociateValue( scalarValue, ScalarVals[0],
				   AssociatedVals[0], Slopes[0] );
	  
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
				   ScalarVals[0], AssociatedVals[0],
				   Slopes[0] );

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

double
Levoy::LightingComponent( const Vector& normal )
{
  double i = ambient_coeff;
  i += diffuse_coeff * Dot( Lvector, normal );

  return i;
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
 * is fixed to be WHITE.
 *
 **************************************************************************/

Color
Levoy::Bresenham2 ( const Point& eye, Vector& step,
	     const Point& beg, const Point& end )
{

  double contribution = 1.0;
  double scalarValue;
  double opacity;
  
  // the color for rendering

  Color colr = WHITE;

  // since no voxels have been crossed, no color has been
  // accumulated.

  Color accumulatedColor( BLACK );

  
  Point voxelB, voxelE;

  // get the voxel indices of beginning and ending points
  
  if ( ! homeSFRGrid->get_voxel( beg, voxelB ) )
    cerr << "1: GET VOXEL RETURNED FALSE!\n";
      
  if ( ! homeSFRGrid->get_voxel( end, voxelE ) )
    cerr << "2: GET VOXEL RETURNED FALSE!\n";

  // perform the initial Bresenham line calculations

  int sdx = voxelE.x() - voxelB.x();
  int sdy = voxelE.y() - voxelB.y();
  int sdz = voxelE.z() - voxelB.z();
  
  int dx = Abs( sdx );
  int dy = Abs( sdy );
  int dz = Abs( sdz );

  int largest = Max( dx, dy, dz );

  int start, stop, decr;
  Vector Primary(0.,0.,0.), Secondary(0.,0.,0.), Tertiary(0.,0.,0.);

  int dS, dT, NES, NET, ES, ET;

  if ( largest == dx )
    {
      start = voxelB.x();
      stop = voxelE.x();
      
      if ( sdx < 0 )
	Primary.x( -1 );
      else
	Primary.x( 1 );

      if ( sdy < 0 )
	Secondary.y( -1 );
      else
	Secondary.y( 1 );

      if ( sdz < 0 )
	Tertiary.z( -1 );
      else
	Tertiary.z( 1 );

      dS = 2 * dy - dx;
      dT = 2 * dz - dx;

      NES = 2 * ( dy - dx );
      NET = 2 * ( dz - dx );

      ES = 2 * dy;
      ET = 2 * dz;
    }
  else if ( largest == dy )
    {
      start = voxelB.y();
      stop = voxelE.y();
      
      if ( sdy < 0 )
	Primary.y( -1 );
      else
	Primary.y( 1 );

      if ( sdx < 0 )
	Secondary.x( -1 );
      else
	Secondary.x( 1 );

      if ( sdz < 0 )
	Tertiary.z( -1 );
      else
	Tertiary.z( 1 );

      dS = 2 * dx - dy;
      dT = 2 * dz - dy;

      NES = 2 * ( dx - dy );
      NET = 2 * ( dz - dy );

      ES = 2 * dx;
      ET = 2 * dz;
    }
  else
    {
      start = voxelB.z();
      stop = voxelE.z();
      
      if ( sdz < 0 )
	Primary.x( -1 );
      else
	Primary.x( 1 );

      if ( sdy < 0 )
	Secondary.y( -1 );
      else
	Secondary.y( 1 );

      if ( sdx < 0 )
	Tertiary.z( -1 );
      else
	Tertiary.z( 1 );

      dS = 2 * dy - dz;
      dT = 2 * dx - dz;

      NES = 2 * ( dy - dz );
      NET = 2 * ( dx - dz );

      ES = 2 * dy;
      ET = 2 * dx;
    }
    
      if ( start < stop )
	decr = 1;
      else
	decr = -1;
      
  Point imat( voxelB );

#if 0  
  cerr << "starting the loop ........" << start << "  " << stop << "\n";
  cerr << "voxel beg: " << voxelB << endl;
  cerr << "voxel end: " << voxelE << endl;
  cerr << dx << "  " << dy << "  " << dz << endl;
  cerr << dS << ": " << ES << ", " << NES << endl;
  cerr << dT << ": " << ET << ", " << NET << endl;
#endif  
  
  while ( contribution > EpsilonContribution && start != stop )
    {
#if 0      
      // TEMP!!! must be removed soon (after good testing)
      if ( ! homeSFRGrid->within_grid( imat ) )
	{
	  cerr << "imat is outside of range\n";
	  cerr << imat << endl;
	  ASSERT( NULL );
	}
#endif      
      
      scalarValue = homeSFRGrid->get_value( imat );

//      cerr << imat << ": The normal is: " << homeSFRGrid->get_normal( imat ) << endl;

      opacity = AssociateValue( scalarValue, ScalarVals[0],
			       AssociatedVals[0], Slopes[0] );

//      opacity *= LightingComponent( homeSFRGrid->get_normal( imat ) );
      
      accumulatedColor += colr * ( contribution * opacity );
      contribution = contribution * ( 1.0 - opacity );

      start += decr;
      
      if ( dS <= 0 )
	dS += ES;
      else
	{
	  dS += NES;
	  imat += Secondary;
	}
      
      if ( dT <= 0 )
	dT += ET;
      else
	{
	  dT += NET;
	  imat += Tertiary;
	}
      
      imat += Primary;
    }
// cerr << endl;

  if ( start == stop && imat != voxelE )
      {
	cerr << "Ending voxel does not correspond to the last voxel\n";
	cerr << "while tracing the ray\n";

	cerr << "ending voxel: " << voxelE << endl;
	cerr << "imat is: "      << imat   << endl;
	
	ASSERT( imat == voxelE );
      }

  // begin traversing through the volume at the first point of
  // intersection of the ray with the bbox.  keep going through
  // the volume until the contribution of the voxel color
  // is almost none, or until the second intersection point has
  // been reached.

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
		  cerr << "HELP!!!!\n\n\n\n\n\n\n\n\n\n\n\nHELP!";
		  ASSERT( ( eye - beg).length() <= ( eye - end ).length() );
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

  // the ratio of near clipping plane to homeRaylength

  nearTohome = g->znear / homeRay.length();

  // rayStep represents the length of vector.  in order to
  // traverse the volume, a vector of this length in the direction
  // of the ray is added to the current point in space

  rayStep = DetermineRayStepSize ();

  // the length of longest possible ray

  dmax = DetermineFarthestDistance ( eye );

  // make sure that raster sizes do not exceed the raster size
  // of the OpenGL window.

  int rx, ry;

  rx = g->xres;
  ry = g->yres;

  if ( g->xres > VIEW_PORT_SIZE )
    g->xres = VIEW_PORT_SIZE;

  if ( g->yres > VIEW_PORT_SIZE )
    g->yres = VIEW_PORT_SIZE;

  cerr << "The new resolutions are: " << g->xres << " " << g->yres << endl;

  // create and initialize the array containing the image

  this->x = g->xres;
  this->y = g->yres;
  
  CharColor temp;


  // deallocate, allocate, and initialize the array
  
  Image->newsize( g->yres, g->xres );
  
  Image->initialize( temp );

  // calculate the increments to be added in the u,v directions

//  CalculateRayIncrements( myview, rayIncrementU, rayIncrementV, x, y );

  CalculateRayIncrements( *(g->view), rayIncrementU, rayIncrementV,
			 g->xres, g->yres );

  cerr << "with new res's: " << rayIncrementU << " " << rayIncrementV << endl;

  CalculateRayIncrements( *(g->view), rayIncrementU, rayIncrementV,
			 rx, ry );

  cerr << "keeping the true res's " << rayIncrementU << " " << rayIncrementV << endl;
  
  // calculate the vector, L, which points toward the sun ( a point
  // light source which is very far away

  Lvector = homeRay + rayIncrementU * 3 + rayIncrementV * 3;
  
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

  // yet another check for whether i calculated the dmax correctly...
  // actually, i probably didn't calculate it correctly...

  if ( mylength + ( eye - beg ).length() + 1.e-5 > dmax )
    {
      cerr << "HELP!!!!\n\n\n\n\n\n\n\n\n\n\n\nHELP!";
      cerr << "DMAX not calculated correctly: " << dmax << " vs " << mylength + ( eye - beg ).length() + 1.e-5 << endl;
      cerr << beg <<"   " << end << endl;
      cerr << "HELP!!!!\n\n\n\n\n\n\n\n\n\n\n\nHELP!";
      ASSERT( NULL );
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
				   AssociatedVals[0], Slopes[0] );

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
	  
	  Color tempcolr( AssociateValue( scalarValue,
					 ScalarVals[1], AssociatedVals[1],
					 Slopes[1] ),
			 AssociateValue( scalarValue,
					ScalarVals[2], AssociatedVals[2],
					Slopes[2] ),
			 AssociateValue( scalarValue,
					ScalarVals[3], AssociatedVals[3],
					Slopes[3] ) );

	  colr = tempcolr;

	  opacity = AssociateValue( scalarValue, ScalarVals[0],
				   AssociatedVals[0], Slopes[0] );
	  
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
				   ScalarVals[0], AssociatedVals[0],
				   Slopes[0] );

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



Array2<CharColor> *
Levoy::BatchTrace ( int projectionType, int repeat )
{
  int loop;
  WallClockTimer watch;

  watch.start();

  for( loop = 0; loop < repeat; loop++ )
    TraceRays( projectionType );

  watch.stop();

  cerr << "The watch reports: " << watch.time() << " for " << repeat <<
    " repetitions\n";
  
  return Image;
}
