#ifndef BRICK_H
#define BRICK_H


#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Containers/Array3.h>
#include <GL/gl.h>
#include <GL/glu.h>

namespace Kurt {
namespace GeomSpace  {


using namespace SCICore::Geometry;
using namespace SCICore::Containers;


class Brick 
{
public:

  Brick(Point min, Point max,
	double alpha,
	bool hasNeighbor,
	const Array3<unsigned char>* tex);

  ~Brick();

  void draw(Ray viewRay, double alpha,
	    double tmin, double tmax, double dt );
  Point getCorner(int i) { return corner[i]; }

  void get_bounds(BBox&);

private:
  typedef struct {
    double base;
    double step;
  } RayStep;

  const Array3<unsigned char>* tex;
  GLuint texName;

  Point corner[8];
  Ray edge[12];
  bool hasNeighbor;
  double alphaScale;
  double aX, aY, aZ;

  void drawSlices(Ray viewRay, double alpha,
		  double tmin, double tmax, double dt);
  void drawWireFrame();

  double ComputeAndDrawPolys(Ray r, double  tmin, double  tmax,
			    double dt, double* ts);
  void OrderIntersects(Point *p, Ray *r, RayStep *dt, int n);
  void drawPolys(Point *intersects, int nIntersects);
  void makeTextureMatrix();

};
#endif


}  // namespace GeomSpace
} // namespace Kurt
