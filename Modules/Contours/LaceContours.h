
/*
 *  LaceContuors.h: Lace a cContourSet into a Surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_LaceContours_h
#define SCI_project_module_LaceContours_h

#include <Dataflow/Module.h>
#include <Datatypes/ContourSet.h>
#include <Datatypes/ContourSetPort.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>

class LaceContours : public Module {
    ContourSetIPort* incontour;
    SurfaceOPort* osurface;

    void lace_contours(const ContourSetHandle&, TriSurface*);

public:
    LaceContours(const clString& id);
    LaceContours(const LaceContours&, int deep);
    virtual ~LaceContours();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
