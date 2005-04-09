/*
 * This class does color and point per vertex
 * it stores everythning explicitly.
 * Peter-Pike Sloan
 */

#ifndef SCI_Geom_QMesh_h
#define SCI_Geom_QMesh_h 1

#include <Geom/Geom.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class GeomQMesh : public GeomObj {
  Array1<float> pts;    // data points
  Array1<float> nrmls;  // normals for above (central differences???)
  Array1<Colorub> clrs; // colors

  int nrows,ncols; // 2d grid of pts...

public:
  GeomQMesh(int, int);
  GeomQMesh(const GeomQMesh&);
  
  virtual ~GeomQMesh();

  virtual GeomObj* clone();

  void add(int, int, Point&, Vector&, Color&); // adds point...
 
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void get_bounds(BBox&);
  virtual void get_bounds(BSphere&);
  virtual void make_prims(Array1<GeomObj*>& free,
			  Array1<GeomObj*>& dontfree);
  virtual void preprocess();
  virtual void intersect(const Ray& ray, Material*,
			 Hit& hit);
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

#endif
