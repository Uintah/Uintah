
/*
 * TriGroup.h:  Group of triangles
 *    (stripped down to improve performance)
 *
 * Packages/Philip Sutton
 * April 1999

  Copyright (C) 2000 SCI Group, University of Utah
 */

#ifndef __GEOM_TRI_GROUP_H__
#define __GEOM_TRI_GROUP_H__

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>

#include <stdio.h>
#include <iostream>

namespace Phil {
using namespace SCIRun;

class pPoint {
public:
  inline pPoint( float x, float y, float z ) : _x(x), _y(y), _z(z) {}
  pPoint() { _x = _y = _z = 0.0; }
  pPoint& operator=( const pPoint& );
  pPoint operator+( const pPoint& ) const;
  pPoint operator-( const pPoint& ) const;
  void x( const float val ) { _x = val; }
  void y( const float val ) { _y = val; }
  void z( const float val ) { _z = val; }
  float x() const { return _x; }
  float y() const { return _y; }
  float z() const { return _z; }
  
  friend inline pPoint Cross( const pPoint&, const pPoint& );
protected:
private:
  float _x, _y, _z;
};

inline 
pPoint& 
pPoint::operator=( const pPoint& p ) {
  _x = p._x;
  _y = p._y;
  _z = p._z;
  return *this;
}

inline
pPoint
pPoint::operator+( const pPoint& p ) const {
  return pPoint( _x+p._x, _y+p._y, _z+p._z );
}

inline
pPoint
pPoint::operator-( const pPoint& p ) const {
  return pPoint( _x-p._x, _y-p._y, _z-p._z );
}

inline
pPoint
Cross( const pPoint& v1, const pPoint& v2 ) {
  return pPoint( 
		v1._y*v2._z-v1._z*v2._y,
		v1._z*v2._x-v1._x*v2._z,
		v1._x*v2._y-v1._y*v2._x);
}

struct Triangle {
  pPoint vertices[3];
  pPoint normal;
};

class GeomTriGroup : public GeomObj {
public:
  GeomTriGroup();
  GeomTriGroup(int n);
  GeomTriGroup(const GeomTriGroup& copy );
  virtual ~GeomTriGroup();

  void add( const pPoint& p1, const pPoint& p2, const pPoint& p3 );
  void shift( double x, double y, double z );
  int getSize() { return size; }
  void write( const char* filename );
  void reserve_clear( int n );

  // inherited from GeomObj
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);

protected:

private:
  Triangle* tris;
  int size;
  int nalloc;
};


} // End namespace Phil



#endif

