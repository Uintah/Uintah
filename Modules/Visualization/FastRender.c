#include <math.h>
#include <stdio.h>
#include "FastRender.h"


#if 1
#define Interpolate(d1, d2, weight) \
((d2)*(weight)+(d1)*(1.0-(weight)))
#else
#define Interpolate(d1, d2, weight) \
(((d2)-(d1))*(weight)+(d1))
#endif

     
     void
     BasicVolRender( double stepx, double stepy, double stepz, double rayStep,
		    double begx, double begy, double begz,
		    double *SVOpacity, double SVmin, double SVMultiplier,
		    double boxx, double boxy, double boxz,
		    double ***grid,
		    double bgr, double bgg, double bgb,
		    int nx, int ny, int nz,
		    double diagx, double diagy, double diagz,
		    double *acr, double *acg, double *acb )
{
  double  EpsilonAlpha = log( 0.01 );

  /* the color for rendering */
  double colrr = 1.0;
  double colrg = 1.0;
  double colrb = 1.0;

  /* as the ray passes through the volume, the Alpha (total opacity)
     value increases */
  double Alpha = 0;

  /* a scalar value corresponds to a particular point
     in space */
  double scalarValue;

  /* opacity for a particular position in the volume */
  double opacity;

  /* position of the point in space at the beginning of the cast */
  double atPointx = begx + 1.0e-6;
  double atPointy = begy + 1.0e-6;
  double atPointz = begz + 1.0e-6;

  /* begin traversing through the volume at the first point of
     intersection of the ray with the bbox.  keep going through
     the volume until the contribution of the voxel color
     is almost none, or until the second intersection point has
     been reached. */

  double z00, z01, z10, z11;
  double y0, y1, inside;
  
  double pnx = atPointx - boxx;
  double pny = atPointy - boxy;
  double pnz = atPointz - boxz;
  
  double dx=stepx*(nx-1)/diagx;
  double dy=stepy*(ny-1)/diagy;
  double dz=stepz*(nz-1)/diagz;

  double x=pnx*(nx-1)/diagx;
  double y=pny*(ny-1)/diagy;
  double z=pnz*(nz-1)/diagz;
  
  int ix=(int)(x+10000)-10000;
  int iy=(int)(y+10000)-10000;
  int iz=(int)(z+10000)-10000;
  
  double fx=x-ix;
  double fy=y-iy;
  double fz=z-iz;
  
  /* JOYJOY */
  double* p=&grid[ix][iy][iz];
  int nynz=ny*nz;
  
  inside = 1;
  
  if(ix<0 || ix+1>=nx)inside=0;
  if(iy<0 || iy+1>=ny)inside=0;
  if(iz<0 || iz+1>=nz)inside=0;
  
  /* accumulated color as the ray passes through the volume
     since no voxels have been crossed, no color has been
     accumulated. */
  *acr = 1.0;
  *acg = 1.0;
  *acb = 1.0;
  scalarValue = 0;
  
  /* the first atPoint may be outside the volume due to roundoff error */
  rayStep*=10;

  if(inside)
    {
      z00=Interpolate(grid[ix][iy][iz], grid[ix][iy][iz+1], fz);
      z01=Interpolate(grid[ix][iy+1][iz], grid[ix][iy+1][iz+1], fz);
      z10=Interpolate(grid[ix+1][iy][iz], grid[ix+1][iy][iz+1], fz);
      z11=Interpolate(grid[ix+1][iy+1][iz], grid[ix+1][iy+1][iz+1], fz);
      y0=Interpolate(z00, z01, fy);
      y1=Interpolate(z10, z11, fy);
      scalarValue=Interpolate(y0, y1, fx);

      opacity = SVOpacity[ (int) ( (scalarValue - SVmin) * SVMultiplier ) ];
      
      Alpha -= opacity * rayStep;
    }

  /* even if initially we were slightly outside, add step and
     then determine if we're inside the bbox */
/*J  inside = 1;*/

  /* JOYJOY */
  SVmin*=SVMultiplier;
  Alpha/=rayStep;
  EpsilonAlpha/=rayStep;
  
  if ( dx >= 0 )
    {
      if ( dy >= 0 )  /* dx>=0 */
	{
	  if ( dz >= 0 )  /* dx>=0, dy>=0 */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }

		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		      
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		      
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }

		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  
		  Alpha -= opacity;
		}
	    }
	  else /* dx>=0, dy>=0, dz<0 */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }

		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue - SVmin) * SVMultiplier ) ];
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }
		  
		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }

		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue - SVmin) * SVMultiplier ) ];
		  Alpha -= opacity;
		}
	    }
	}
      else  /* dx>=0   dy < 0 */
	{
	  if ( dz >= 0 ) /* dx>=0, dy<0, dz>=0 */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }
		  
		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		      
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }
		  
		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		}
	    }
	  else /* dx>=0, dy<0, dz<0 */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }

		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		      
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx>=1){
		    do {
		      fx-=1.0;
		      ix++;
		      p+=nynz;
		    } while(fx>=1);
		    if ( ix+1 >= nx )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }

		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		}
	    }
	}
    }
  else   /* dx < 0 */
    {
      if ( dy >= 0 ) /* dx<0 */
	{
	  if ( dz >= 0 )  /* dx<0, dy>=0 */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;

		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		      
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		}
	    }
	  else  /* dz < 0  (dx<0,dy>=0) */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;

		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		      
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy>=1){
 		    do {
		      fy-=1.0;
		      iy++;
		      p+=nz;
		    } while(fy>=1);
		    if ( iy+1 >= ny )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		}
	    }
	}
      else  /* dx<0    dy < 0 */
	{
	  if ( dz >= 0)  /* dx<0, dy<0 */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;

		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }
		  
		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		      
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fz>=1){
		    do {
		      fz-=1.0;
		      iz++;
		      p++;
		    } while(fz>=1);
		    if( iz+1 >= nz )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		}
	    }
	  else  /* dx,dy,dz < 0 */
	    {
	      /*********************************************************
	       *********************************************************/
	      while ( Alpha >= EpsilonAlpha )
		{ 
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;

		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }
		  
		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		      
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		  
		  fx+=dx;
		  fy+=dy;
		  fz+=dz;
		  
		  if(fx<0){
		    do {
		      fy+=1.0;
		      ix--;
		      p+=nynz;
		    } while(fx<0);
		    if ( iy < 0 )
		      break;
		  }

		  if(fy<0){
 		    do {
		      fy+=1.0;
		      iy--;
		      p+=nz;
		    } while(fy<0);
		    if ( iy < 0 )
		      break;
		  }
		  
		  if(fz<0){
		    do {
		      fz+=1.0;
		      iz--;
		      p++;
		    } while(fz<0);
		    if( iz < 0 )
		      break;
		  }

		  z00=Interpolate(p[0], p[1], fz);
		  z01=Interpolate(p[nz], p[nz+1], fz);
		  z10=Interpolate(p[nynz], p[nynz+1], fz);
		  z11=Interpolate(p[nynz+nz], p[nynz+nz+1], fz);
		  y0=Interpolate(z00, z01, fy);
		  y1=Interpolate(z10, z11, fy);
		  scalarValue=Interpolate(y0, y1, fx);
		  
		  opacity = SVOpacity[(int)( (scalarValue * SVMultiplier ) - SVmin)];
		  Alpha -= opacity;
		}
	    }
	}
    }

#if 0  
  while ( Alpha >= EpsilonAlpha   &&   inside )
    { 
      fx+=dx;
      fy+=dy;
      fz+=dz;
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
	  
	  opacity = SVOpacity[(int)( (scalarValue - SVmin) * SVMultiplier ) ];

	  Alpha -= opacity * rayStep;
	}
    }
#endif

  if ( Alpha <= EpsilonAlpha )
    printf(":");
  
  Alpha*=rayStep;
  Alpha = exp( Alpha );
  
  /* background color should contribute to the final color of
     the pixel. */

  *acr = (1-Alpha) + bgr * Alpha;
  *acg = (1-Alpha) + bgg * Alpha;
  *acb = (1-Alpha) + bgb * Alpha;
}

void
ColorVolRender( double stepx, double stepy, double stepz, double rayStep,
	       double begx, double begy, double begz,
	       double *SVOpacity, double *SVR, double *SVG, double *SVB,
	       double SVmin, double SVMultiplier,
	       double boxx, double boxy, double boxz,
	       double ***grid,
	       double bgr, double bgg, double bgb,
	       int nx, int ny, int nz,
	       double diagx, double diagy, double diagz,
	       double *acr, double *acg, double *acb )
{
  /* a scalar value corresponds to a particular point
     in space */
  double scalarValue;

  /* opacity for a particular position in the volume */
  double opacity;

  double  EpsilonAlpha = log( 0.01 );

  /* the color for rendering */
  double colrr = 1.0;
  double colrg = 1.0;
  double colrb = 1.0;

  /* as the ray passes through the volume, the Alpha (total opacity)
     value increases */
  double Alphar = 0.0;
  double Alphag = 0.0;
  double Alphab = 0.0;

  /* position of the point in space at the beginning of the cast */
  double atPointx = begx + 1.0e-6;
  double atPointy = begy + 1.0e-6;
  double atPointz = begz + 1.0e-6;


  /* begin traversing through the volume at the first point of
     intersection of the ray with the bbox.  keep going through
     the volume until the contribution of the voxel color
     is almost none, or until the second intersection point has
     been reached. */

  double z00, z01, z10, z11;
  double y0, y1, inside;
  
  double pnx = atPointx - boxx;
  double pny = atPointy - boxy;
  double pnz = atPointz - boxz;
  
  double dx=stepx*(nx-1)/diagx;
  double dy=stepy*(ny-1)/diagy;
  double dz=stepz*(nz-1)/diagz;

  double x=pnx*(nx-1)/diagx;
  double y=pny*(ny-1)/diagy;
  double z=pnz*(nz-1)/diagz;
  
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
  
  /* accumulated color as the ray passes through the volume
     since no voxels have been crossed, no color has been
     accumulated. */
  *acr = 1.0;
  *acg = 1.0;
  *acb = 1.0;
  scalarValue = 0;

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

      opacity = SVOpacity[ (int) ( (scalarValue - SVmin) * SVMultiplier ) ];
      colrr = SVR[ (int)( (scalarValue - SVmin) * SVMultiplier ) ];
      colrg = SVG[ (int)( (scalarValue - SVmin) * SVMultiplier ) ];
      colrb = SVB[ (int)( (scalarValue - SVmin) * SVMultiplier ) ];
      
      Alphar = Alphar - opacity * rayStep / 0.1 * colrr;
      Alphag = Alphag - opacity * rayStep / 0.1 * colrg;
      Alphab = Alphab - opacity * rayStep / 0.1 * colrb;
    }

  /* even if initially we were slightly outside, add step and
     then determine if we're inside the bbox */
  inside = 1;
  
  while ( ( Alphar >= EpsilonAlpha || Alphag >= EpsilonAlpha ||
	   Alphab >= EpsilonAlpha )   &&   inside )
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

	  opacity = SVOpacity[ (int) ( (scalarValue - SVmin) * SVMultiplier ) ];
	  colrr = SVR[ (int)( (scalarValue - SVmin) * SVMultiplier ) ];
	  colrg = SVG[ (int)( (scalarValue - SVmin) * SVMultiplier ) ];
	  colrb = SVB[ (int)( (scalarValue - SVmin) * SVMultiplier ) ];
	  
	  Alphar = Alphar - opacity * rayStep / 0.1 * colrr;
	  Alphag = Alphag - opacity * rayStep / 0.1 * colrg;
	  Alphab = Alphab - opacity * rayStep / 0.1 * colrb;
	}
    }

  
  Alphar = exp( Alphar );
  Alphag = exp( Alphag );
  Alphab = exp( Alphab );
  
  /* background color should contribute to the final color of
     the pixel.
     accumulatedColor = WHITE * ( 1 - Alpha ) + backgroundColor * Alpha;
     somewhat INCORRECT composition of the background color and Alpha...
     */

  *acr = 1 - Alphar + bgr * Alphar;
  *acg = 1 - Alphag + bgg * Alphag;
  *acb = 1 - Alphab + bgb * Alphab;
  /*
     accumulatedColor.r( ( 1 - Alpha.r() ) + backgroundColor.r() * Alpha.r() );
     accumulatedColor.g( ( 1 - Alpha.g() ) + backgroundColor.g() * Alpha.g() );
     accumulatedColor.b( ( 1 - Alpha.b() ) + backgroundColor.b() * Alpha.b() );
     */
}
