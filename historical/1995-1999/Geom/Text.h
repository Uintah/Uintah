

/*
 *  Text.h:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#ifndef SCI_Geom_Text_h
#define SCI_Geom_Text_h 1

#include <Geom/Geom.h>
#include <GL/gl.h>

class GeomText : public GeomObj {
  static int init;
  static GLuint fontbase;
public:
  clString text;
  Point at;
  Color c;
public:
  GeomText();
  GeomText( const clString &, const Point &, const Color &c = Color(1,1,1));
  GeomText(const GeomText&);
  virtual ~GeomText();
  virtual GeomObj* clone();

  virtual void reset_bbox();
  virtual void get_bounds(BBox&);
  virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};


#endif /* SCI_Geom_Text_h */

