
/*
 *  MeshView.h: The 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_MeshView_h
#define SCI_project_module_MeshView_h

#include <UserModule.h>
#include <Geometry/Point.h>
#include <MUI.h>
#include "connect.h"

class ColormapPort;
class Field3DIPort;
class GeometryOPort;
class ObjGroup;

class MeshView : public UserModule {
    GeometryOPort* ogeom;
    int abort_flag;
    MUI_slider_int *levSlide;
    
    int numLevels, oldLev;
    int seedTet, oldSeed;
    double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    double clipX, oldClipX, clipY, oldClipY, clipZ, oldClipZ;
    int deep;
    int allLevels;

    int widget_id;
    double *data;
    int *tetra;
    
    int numVerts;
    int numTetra;

    LPTR *list;
    int	*levels;

public:
    MeshView();
    MeshView(const MeshView&, int deep);
    virtual ~MeshView();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
    void initList();
    LPTR newList();
    void addTet(int row, int ind);
    void makeLevels();
    void readDat();
};

#endif






