
/*
 *  GeomTimeGroup.h:  ?
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Time__Group_h
#define SCI_Geom_Time_Group_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

class SCICORESHARE GeomTimeGroup : public GeomObj {
    Array1<GeomObj*> objs;
    Array1<double>   start_times;
    int del_children;

    BBox bbox; // bbox for entire seen - set once!
public:
    GeomTimeGroup(int del_children=1);
    GeomTimeGroup(const GeomTimeGroup&);
    virtual ~GeomTimeGroup();
    virtual GeomObj* clone();

    void add(GeomObj*,double); // with time...
    void remove(GeomObj*);
    void remove_all();
    int size();

    void  setbbox(BBox&); // sets bounding box - so isn't computed!

    virtual void reset_bbox();
    virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
};

} // End namespace SCIRun


#endif /* SCI_Geom_Group_h */
