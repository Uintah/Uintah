
/*
 *  IndexedGroup.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef SCI_Geom_IndexedGroup_h
#define SCI_Geom_IndexedGroup_h 1

#include <Containers/Array1.h>
#include <Containers/HashTable.h>

#include <Geom/GeomObj.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>

namespace SCICore {
namespace GeomSpace {

using SCICore::Containers::HashTable;
using SCICore::Containers::HashTableIter;

class GeomIndexedGroup: public GeomObj {
    HashTable<int, GeomObj*> objs;
public:
    GeomIndexedGroup( const GeomIndexedGroup& );
    GeomIndexedGroup();
    virtual ~GeomIndexedGroup();

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

    void addObj(GeomObj*, int);  // adds an object to the table
    GeomObj* getObj(int);        // gets an object from the table
    void delObj(int, int del);	 // removes object from table
    void delAll(void);		 // deletes everything

    HashTableIter<int,GeomObj*> getIter(void); // gets an iter
    HashTable<int,GeomObj*>* getHash(void);    // gets the table
    virtual bool saveobj(ostream&, const clString& format, GeomSave*);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:49  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:11  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:08  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif
