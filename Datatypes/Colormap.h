
/*
 *  Colormap.h: Colormap definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Colormap_h
#define SCI_project_Colormap_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/LockingHandle.h>
#include <Geom/Material.h>

class Colormap;
typedef LockingHandle<Colormap> ColormapHandle;

class Colormap : public Datatype {
public:
    Array1<MaterialHandle> colors;

    Colormap(int nlevels);

    virtual ~Colormap();
    virtual Colormap* Colormap::clone();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
