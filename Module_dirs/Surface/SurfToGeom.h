
/*
 *  SurfToGeom.h: Convert a surface into geometry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_SurfToGeom_h
#define SCI_project_module_SurfToGeom_h

#include <Dataflow/Module.h>
#include <Datatypes/ContourSet.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Geom/Geom.h>

class SurfToGeom : public Module {
    SurfaceIPort* isurface;
    GeometryOPort* ogeom;

    void surf_to_geom(const SurfaceHandle&, ObjGroup*);

public:
    SurfToGeom(const clString& id);
    SurfToGeom(const SurfToGeom&, int deep);
    virtual ~SurfToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
