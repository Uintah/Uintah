#ifndef TEXBRICK_H
#define TEXBRICK_H


#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Geom/GeomObj.h>
#include <GL/gl.h>
#include <GL/glu.h>

namespace SCICore {
namespace GeomSpace  {


using namespace SCICore::Geometry;


class TexBrick : public GeomObj
{
public:

  TexBrick(int id, int slices, double alpha,
	   bool hasNeighbor,
	   Point min, Point max,
	   int xoff, int yoff, int zoff,
	   int texX, int texY, int texZ,
	   int padx, int pady, int padz,
	   const GLvoid* tex,
	   const GLvoid* cmap);

  TexBrick(const TexBrick&);
  ~TexBrick();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);

private:
  void drawSlices();
  void drawWireFrame();
  // dt = step size
  // will eventually be private
  void computeParameters( const Vector& N, const Ray& viewRay,
			  float *t, int len_t);
  void sortParameters( float *t, int len_t);

  Point getCorner(int i) { return corner[i]; }

  typedef struct {
    double base;
    double step;
  } RayStep;

  const GLvoid *tex;
  const GLvoid *cmap;
  
  int X,Y,Z;  // size of the texture
  int xoff, yoff, zoff;
  int padx, pady, padz;
  int slices;
  Point corner[8];
  Ray edge[12];
  bool hasNeighbor;
  double alpha;

  float aX, aY, aZ;

  float computeParameter( const Vector& normal, const Point& corner,  
			  const Ray& ray );
  float ComputeAndDrawPolys(Ray r, float tmin, float tmax,
			    float dt, float* ts);
  void OrderIntersects(Point *p, Ray *r, RayStep *dt, int n);
  void drawPolys(Point *intersects, int nIntersects);
  void makeTextureMatrix();

  GLuint texName;

};
#endif


}  // namespace GeomSpace
} // namespace SCICore
