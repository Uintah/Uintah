
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

#include <UserModule.h>
#include <ContourSet.h>
#include <Surface.h>
#include <SurfacePort.h>
#include <Geom.h>
#include <GeometryPort.h>

class SurfToGeom : public UserModule {
    SurfaceIPort* isurface;
    GeometryOPort* ogeom;

    void surf_to_geom(const SurfaceHandle&, ObjGroup*);

public:
    SurfToGeom();
    SurfToGeom(const SurfToGeom&, int deep);
    virtual ~SurfToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
//    virtual void mui_callback(void*, int);
};

#endif
