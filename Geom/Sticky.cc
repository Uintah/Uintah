#include "Sticky.h"
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <Geometry/BBox.h>

Persistent *make_GeomSticky() {
  return scinew GeomSticky( 0 );
}

PersistentTypeID GeomSticky::type_id("GeomSticky", "GeomObj", make_GeomSticky);

GeomSticky::GeomSticky( GeomObj *c )
  : GeomObj(), child(c)
{
}

GeomSticky::GeomSticky( const GeomSticky &copy )
  : GeomObj(copy), child(copy.child)
{
}

GeomSticky::~GeomSticky()
{
  if(child)
    delete child;
}

GeomObj* GeomSticky::clone() {
  return scinew GeomSticky( *this );
}

void GeomSticky::get_bounds( BBox& bb ) {
  child->get_bounds( bb );
}

void GeomSticky::get_bounds(BSphere& sp) {
  child->get_bounds( sp );
}

void GeomSticky::make_prims( Array1<GeomObj*>&, Array1<GeomObj*>&)
{
  NOT_FINISHED("GeomSticky::make_prims");
}

void GeomSticky::preprocess() {
  NOT_FINISHED("GeomSticky::preprocess");
}

void GeomSticky::intersect(const Ray& ray, Material*, Hit& hit) {
  NOT_FINISHED("GeomSticky::intersect");
}

#define GeomSticky_VERSION 1

void GeomSticky::io(Piostream& stream) {
  stream.begin_class("GeomSticky", GeomSticky_VERSION);
  GeomObj::io(stream);
  Pio(stream, child);
  stream.end_class();
}

bool GeomSticky::saveobj(ostream&, const clString& format, GeomSave*) {
  NOT_FINISHED("GeomSticky::saveobj");
  return false;
}


