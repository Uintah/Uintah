#include "FastRender.h"

Vector EpsilonVector( 1.e-6, 1.e-6, 1.e-6 );
double  EpsilonAlpha = log( 0.01 );

const Color BLACK( 0., 0., 0. );
const Color WHITE( 1., 1., 1. );


Color
BasicVolRender( Vector& step, double rayStep, const Point& beg,
		 double *SVOpacity, double SVmin, double SVMultiplier,
	       const BBox& box, double ***grid, Color& backgroundColor,
	       int nx, int ny, int nz, int diagx, int diagy, int diagz )
{
  /* the color for rendering */
  Color colr = WHITE;

  /* find step vector */
  step *= rayStep;

  /* as the ray passes through the volume, the Alpha (total opacity)
   value increases */
  double Alpha = 0;

  /* accumulated color as the ray passes through the volume
   since no voxels have been crossed, no color has been
   accumulated. */
  Color accumulatedColor = BLACK;

  /* a scalar value corresponds to a particular point
    in space */
  double scalarValue;

  /* opacity for a particular position in the volume */
  double opacity;

  /* position of the point in space at the beginning of the cast */
  Point atPoint = beg + EpsilonVector;

  /* begin traversing through the volume at the first point of
   intersection of the ray with the bbox.  keep going through
   the volume until the contribution of the voxel color
   is almost none, or until the second intersection point has
   been reached. */
  scalarValue = 0;

  double z00, z01, z10, z11;
  double y0, y1, inside;
  
  Vector pn=atPoint-box.min();
  
  double dx=step.x()*(nx-1)/diagx;
  double dy=step.y()*(ny-1)/diagy;
  double dz=step.z()*(nz-1)/diagz;

  double x=pn.x()*(nx-1)/diagx;
  double y=pn.y()*(ny-1)/diagy;
  double z=pn.z()*(nz-1)/diagz;
  
  int ix=(int)(x+10000)-10000;
  int iy=(int)(y+10000)-10000;
  int iz=(int)(z+10000)-10000;
  
  double fx=x-ix;
  double fy=y-iy;
  double fz=z-iz;
  
  inside = 1;
  
  if(ix<0 || ix+1>=nx)inside=0;
  if(iy<0 || iy+1>=ny)inside=0;
  if(iz<0 || iz+1>=nz)inside=0;
  
  /* the first atPoint may be outside the volume due to roundoff error */

  if(inside)
    {

      z00=Interpolate(grid[ix][iy][iz], grid[ix][iy][iz+1], fz);
      z01=Interpolate(grid[ix][iy+1][iz], grid[ix][iy+1][iz+1], fz);
      z10=Interpolate(grid[ix+1][iy][iz], grid[ix+1][iy][iz+1], fz);
      z11=Interpolate(grid[ix+1][iy+1][iz], grid[ix+1][iy+1][iz+1], fz);
      y0=Interpolate(z00, z01, fy);
      y1=Interpolate(z10, z11, fy);
      scalarValue=Interpolate(y0, y1, fx);

      opacity = SVOpacity[ int( (scalarValue - SVmin) * SVMultiplier) ];
      
      Alpha += - opacity * rayStep / 0.1;
    }

  /* even if initially we were slightly outside, add step and
   then determine if we're inside the bbox */
  inside = 1;
  
  while ( Alpha >= EpsilonAlpha   &&   inside )
    { 
      fx+=dx;
      if(fx<0){
	int i=(int)(-fx)+1;
	fx+=i;
	ix-=i;
	if(ix<0)
	  inside=0;
      } else if(fx>=1){
	int i=(int)fx;
	fx-=i;
	ix+=i;
	if(ix+1 >= nx)
	  inside=0;
      }
      
      fy+=dy;
      if(fy<0){
	int i=(int)(-fy)+1;
	fy+=i;
	iy-=i;
	if(iy<0)
	  inside=0;
      } else if(fy>=1){
	int i=(int)fy;
	fy-=i;
	iy+=i;
	if(iy+1 >= ny)
	  inside=0;
      }
      
      fz+=dz;
      if(fz<0){
	int i=(int)(-fz)+1;
	fz+=i;
	iz-=i;
	if(iz<0)
	  inside=0;
      } else if(fz>=1){
	int i=(int)fz;
	fz-=i;
	iz+=i;
	if(iz+1 >= nz)
	  inside=0;
      }

      if(inside)
	{
	  
	  z00=Interpolate(grid[ix][iy][iz], grid[ix][iy][iz+1], fz);
	  z01=Interpolate(grid[ix][iy+1][iz], grid[ix][iy+1][iz+1], fz);
	  z10=Interpolate(grid[ix+1][iy][iz], grid[ix+1][iy][iz+1], fz);
	  z11=Interpolate(grid[ix+1][iy+1][iz], grid[ix+1][iy+1][iz+1], fz);
	  y0=Interpolate(z00, z01, fy);
	  y1=Interpolate(z10, z11, fy);
	  scalarValue=Interpolate(y0, y1, fx);
	  
	  opacity = SVOpacity[ int( (scalarValue - SVmin) * SVMultiplier) ];

	  Alpha += - opacity * rayStep / 0.1;
	}
    }

  
  Alpha = exp( Alpha );
  
  /* background color should contribute to the final color of
   the pixel. */
  
  accumulatedColor = WHITE * ( 1 - Alpha ) + backgroundColor * Alpha;

  return accumulatedColor;
}


Color
ColorVolRender( Vector& step, double rayStep, const Point& beg,
	       double *SVOpacity, double *SVR, double *SVG, double *SVB,
	       double SVmin, double SVMultiplier,
	       const BBox& box, double ***grid, Color& backgroundColor,
	       int nx, int ny, int nz, int diagx, int diagy, int diagz )
{
  /* the color for rendering */
  Color colr = WHITE;

  /* find step vector */
  step *= rayStep;

  /* as the ray passes through the volume, the Alpha (total opacity)
   value increases */
  Color Alpha = BLACK;

  /* accumulated color as the ray passes through the volume
   since no voxels have been crossed, no color has been
   accumulated. */
  Color accumulatedColor = BLACK;

  /* a scalar value corresponds to a particular point
    in space */
  double scalarValue;

  /* opacity for a particular position in the volume */
  double opacity;

  /* position of the point in space at the beginning of the cast */
  Point atPoint = beg + EpsilonVector;

  /* begin traversing through the volume at the first point of
   intersection of the ray with the bbox.  keep going through
   the volume until the contribution of the voxel color
   is almost none, or until the second intersection point has
   been reached. */
  scalarValue = 0;

  double z00, z01, z10, z11;
  double y0, y1, inside;
  
  Vector pn=atPoint-box.min();
  
  double dx=step.x()*(nx-1)/diagx;
  double dy=step.y()*(ny-1)/diagy;
  double dz=step.z()*(nz-1)/diagz;

  double x=pn.x()*(nx-1)/diagx;
  double y=pn.y()*(ny-1)/diagy;
  double z=pn.z()*(nz-1)/diagz;
  
  int ix=(int)(x+10000)-10000;
  int iy=(int)(y+10000)-10000;
  int iz=(int)(z+10000)-10000;
  
  double fx=x-ix;
  double fy=y-iy;
  double fz=z-iz;
  
  inside = 1;
  
  if(ix<0 || ix+1>=nx)inside=0;
  if(iy<0 || iy+1>=ny)inside=0;
  if(iz<0 || iz+1>=nz)inside=0;
  
  /* the first atPoint may be outside the volume due to roundoff error */
  if(inside)
    {

      z00=Interpolate(grid[ix][iy][iz], grid[ix][iy][iz+1], fz);
      z01=Interpolate(grid[ix][iy+1][iz], grid[ix][iy+1][iz+1], fz);
      z10=Interpolate(grid[ix+1][iy][iz], grid[ix+1][iy][iz+1], fz);
      z11=Interpolate(grid[ix+1][iy+1][iz], grid[ix+1][iy+1][iz+1], fz);
      y0=Interpolate(z00, z01, fy);
      y1=Interpolate(z10, z11, fy);
      scalarValue=Interpolate(y0, y1, fx);
      
      opacity = SVOpacity[ int( (scalarValue - SVmin) * SVMultiplier) ];
      colr.r( SVR[ int( (scalarValue - SVmin) * SVMultiplier ) ] );
      colr.g( SVG[ int( (scalarValue - SVmin) * SVMultiplier ) ] );
      colr.b( SVB[ int( (scalarValue - SVmin) * SVMultiplier ) ] );
      
      Alpha.r( Alpha.r() - opacity * rayStep / 0.1 * colr.r() );
      Alpha.g( Alpha.g() - opacity * rayStep / 0.1 * colr.g() );
      Alpha.b( Alpha.b() - opacity * rayStep / 0.1 * colr.b() );
    }

  /* even if initially we were slightly outside, add step and
   then determine if we're inside the bbox */
  inside = 1;
  
  while ( ( Alpha.r() >= EpsilonAlpha || Alpha.g() >= EpsilonAlpha ||
	   Alpha.b() >= EpsilonAlpha )   &&   inside )
    { 
      fx+=dx;
      if(fx<0){
	int i=(int)(-fx)+1;
	fx+=i;
	ix-=i;
	if(ix<0)
	  inside=0;
      } else if(fx>=1){
	int i=(int)fx;
	fx-=i;
	ix+=i;
	if(ix+1 >= nx)
	  inside=0;
      }
      
      fy+=dy;
      if(fy<0){
	int i=(int)(-fy)+1;
	fy+=i;
	iy-=i;
	if(iy<0)
	  inside=0;
      } else if(fy>=1){
	int i=(int)fy;
	fy-=i;
	iy+=i;
	if(iy+1 >= ny)
	  inside=0;
      }
      
      fz+=dz;
      if(fz<0){
	int i=(int)(-fz)+1;
	fz+=i;
	iz-=i;
	if(iz<0)
	  inside=0;
      } else if(fz>=1){
	int i=(int)fz;
	fz-=i;
	iz+=i;
	if(iz+1 >= nz)
	  inside=0;
      }

      if(inside)
	{
	  
	  z00=Interpolate(grid[ix][iy][iz], grid[ix][iy][iz+1], fz);
	  z01=Interpolate(grid[ix][iy+1][iz], grid[ix][iy+1][iz+1], fz);
	  z10=Interpolate(grid[ix+1][iy][iz], grid[ix+1][iy][iz+1], fz);
	  z11=Interpolate(grid[ix+1][iy+1][iz], grid[ix+1][iy+1][iz+1], fz);
	  y0=Interpolate(z00, z01, fy);
	  y1=Interpolate(z10, z11, fy);
	  scalarValue=Interpolate(y0, y1, fx);
	  
	  opacity = SVOpacity[ int( (scalarValue - SVmin) * SVMultiplier) ];
	  colr.r( SVR[ int( (scalarValue - SVmin) * SVMultiplier ) ] );
	  colr.g( SVG[ int( (scalarValue - SVmin) * SVMultiplier ) ] );
	  colr.b( SVB[ int( (scalarValue - SVmin) * SVMultiplier ) ] );
	  
	  Alpha.r( Alpha.r() - opacity * rayStep / 0.1 * colr.r() );
	  Alpha.g( Alpha.g() - opacity * rayStep / 0.1 * colr.g() );
	  Alpha.b( Alpha.b() - opacity * rayStep / 0.1 * colr.b() );
	}
    }

  
  Alpha.r( exp( Alpha.r() ) );
  Alpha.g( exp( Alpha.g() ) );
  Alpha.b( exp( Alpha.b() ) );
  
  /* background color should contribute to the final color of
   the pixel.
  accumulatedColor = WHITE * ( 1 - Alpha ) + backgroundColor * Alpha;
  somewhat INCORRECT composition of the background color and Alpha...
  */
  
  accumulatedColor.r( ( 1 - Alpha.r() ) + backgroundColor.r() * Alpha.r() );
  accumulatedColor.g( ( 1 - Alpha.g() ) + backgroundColor.g() * Alpha.g() );
  accumulatedColor.b( ( 1 - Alpha.b() ) + backgroundColor.b() * Alpha.b() );

  return accumulatedColor;
}
