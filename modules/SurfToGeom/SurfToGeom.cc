
/*
 *  SurfToGeom.cc:  Convert a surface into geoemtry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SurfToGeom/SurfToGeom.h>
#include <Surface.h>
#include <SurfacePort.h>
#include <Geom.h>
#include <GeometryPort.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_SurfToGeom()
{
    return new SurfToGeom;
}

static RegisterModule db1("Surface", "Surface to Geometry", make_SurfToGeom);
static RegisterModule db2("Visualization", "Surface to Geometry", 
			  make_SurfToGeom);

SurfToGeom::SurfToGeom()
: UserModule("SurfToGeom", Filter)
{
    // Create the input port
    isurface=new SurfaceIPort(this, "Surface", SurfaceIPort::Atomic);
    add_iport(isurface);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

SurfToGeom::SurfToGeom(const SurfToGeom&copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("SurfToGeom::SurfToGeom");
}

SurfToGeom::~SurfToGeom()
{
}

Module* SurfToGeom::clone(int deep)
{
    return new SurfToGeom(*this, deep);
}

void SurfToGeom::execute()
{
    SurfaceHandle surf;
    if (!isurface->get(surf))
	return;
    ObjGroup *group=surf->getGeomFromSurface();
    ogeom->delAll();
    ogeom->addObj(group, surf->name);
}
