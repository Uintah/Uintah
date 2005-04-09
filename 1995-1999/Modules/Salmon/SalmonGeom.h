#ifndef SCI_Salmon_Geom_h
#define SCI_Salmon_Geom_h 1

#include <Geom/IndexedGroup.h>
#include <Classlib/String.h>

class GeometryComm;
class CrowdMonitor;

/* this is basicaly a indexed group that also has some simple message
 * stuff
 */	

class GeomSalmonPort: public GeomIndexedGroup {
    GeometryComm* msg_head;
    GeometryComm* msg_tail;

    int portno;
    
public:
    friend class Salmon;
    GeomSalmonPort(int);
    virtual ~GeomSalmonPort();

    GeometryComm* getHead(void) { return msg_head; }
};

/*
 * for items in a scene - has name (for roes to lookup)
 * a lock and a geomobj (to call)
 */

class GeomSalmonItem: public GeomObj {
    GeomObj *child;
    clString name;
    CrowdMonitor* lock;

public:
    friend class Roe;
    GeomSalmonItem();
    GeomSalmonItem(GeomObj*,const clString&, CrowdMonitor* lock);
    virtual ~GeomSalmonItem();

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
    bool saveobj(ostream& out, const clString& format,
		 GeomSave* saveinfo);
    
    clString& getString(void) { return name;}
};

#endif
