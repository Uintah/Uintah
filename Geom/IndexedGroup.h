
#ifndef SCI_Geom_IndexedGroup_h
#define SCI_Geom_IndexedGroup_h 1

#include <Classlib/Array1.h>
#include <Classlib/HashTable.h>

#include <Geom/Geom.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>

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
};

#endif
