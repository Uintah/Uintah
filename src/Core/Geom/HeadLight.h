
/*
 *  HeadLight.h:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_HeadLight_h
#define SCI_Geom_HeadLight_h 1

#include <Core/share/share.h>

#include <Core/Geom/Light.h>
#include <Core/Geom/Color.h>

namespace SCIRun {


class SCICORESHARE HeadLight : public Light {
    Color c;
public:
    HeadLight(const clString& name, const Color&);
    virtual ~HeadLight();

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
};

} // End namespace SCIRun


#endif /* SCI_Geom_HeadLight_h */
