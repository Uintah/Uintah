#ifndef _FAST_RENDER_H
#define _FAST_RENDER_H 1

#ifdef __cplusplus
extern "C" {
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
	       double *acr, double *acg, double *acb );

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
	       double *acr, double *acg, double *acb );

#ifdef __cplusplus
}
#endif
	       
#endif
