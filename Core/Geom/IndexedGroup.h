
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

#include <Core/Containers/Array1.h>
#include <map.h>

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomIndexedGroup: public GeomObj {
    
public:
    typedef map< int, GeomObj*, less<int> > MapIntGeomObj;
    
    typedef pair< MapIntGeomObj::iterator,
                  MapIntGeomObj::iterator >
        IterIntGeomObj;

private:
    
    MapIntGeomObj objs;
    
public:

    GeomIndexedGroup( const GeomIndexedGroup& );
    GeomIndexedGroup();
    virtual ~GeomIndexedGroup();

    virtual GeomObj* clone();
    virtual void reset_bbox();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;

    void addObj(GeomObj*, int);  // adds an object to the table
    GeomObj* getObj(int);        // gets an object from the table
    void delObj(int, int del);	 // removes object from table
    void delAll(void);		 // deletes everything

    IterIntGeomObj getIter(void); // gets an iter 
    
    MapIntGeomObj* getHash(void); // gets the table
    
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun


#endif
