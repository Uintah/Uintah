
/*
 *  LaceContuors.h: Lace a cContourSet into a Surface
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_LaceContours_h
#define SCI_project_module_LaceContours_h

#include <UserModule.h>
#include <ContourSet.h>
#include <Surface.h>
#include <SurfacePort.h>
#include <ContourSetPort.h>

class LaceContours : public UserModule {
    ContourSetIPort* incontour;
    SurfaceOPort* osurface;

    void lace_contours(const ContourSetHandle&, TriSurface*);

public:
    LaceContours();
    LaceContours(const LaceContours&, int deep);
    virtual ~LaceContours();
    virtual Module* clone(int deep);
    virtual void execute();
//    virtual void mui_callback(void*, int);
};

#endif
