
/*
 *  Light.h: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Light_h
#define SCI_Geom_Light_h 1

#include <Core/share/share.h>

#ifndef _WIN32
#include <sci_config.h>
#endif
#include <Core/Persistent/Persistent.h>
#include <Core/Containers/String.h>

namespace SCIRun {

class Point;
class Vector;
class Color;
class GeomObj;
class OcclusionData;
class View;
struct DrawInfoOpenGL;


class SCICORESHARE Light : public Persistent {
protected:
    Light(const clString& name);
public:
    clString name;
    virtual ~Light();
    virtual void io(Piostream&);

    friend SCICORESHARE void Pio( Piostream&, Light*& );

    static PersistentTypeID type_id;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)=0;
#endif
};

} // End namespace SCIRun


#endif /* SCI_Geom_Light_h */

