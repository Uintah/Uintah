
/*
 *  PointLight.h:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_PointLight_h
#define SCI_Geom_PointLight_h 1

#include <Core/Geom/Light.h>
#include <Core/Geom/Color.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SCICORESHARE PointLight : public Light {
    Point p;
    Color c;
public:
    PointLight(const clString& name, const Point&, const Color&);
    virtual ~PointLight();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
};

} // End namespace SCIRun


#endif /* SCI_Geom_PointLight_h */

