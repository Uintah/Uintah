#ifndef _FAST_RENDER_H
#define _FAST_RENDER_H 1

#include <Geom/Color.h>
#include <Geometry/Vector.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>

     
Color
BasicVolRender( Vector& step, double rayStep, const Point& beg,
		 double *SVOpacity, double SVmin, double SVMultiplier,
	       const BBox& box, double ***grid, Color& backgroundColor,
	       int nx, int ny, int nz, int diagx, int diagy, int diagz );

Color
ColorVolRender( Vector& step, double rayStep, const Point& beg,
	       double *SVOpacity, double *SVR, double *SVG, double *SVB,
	       double SVmin, double SVMultiplier,
	       const BBox& box, double ***grid, Color& backgroundColor,
	       int nx, int ny, int nz, int diagx, int diagy, int diagz );

#endif
