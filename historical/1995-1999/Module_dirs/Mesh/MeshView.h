
/*
 *  MeshView.h: An experimental Mesh visualizer
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

#include <Dataflow/Module.h>
#include <Datatypes/MeshPort.h>
#include <Geometry/Point.h>
#include <Geom/Geom.h>
#include "connect.h"

class GeometryOPort;
class Mesh;
class GeomGroup;

class MeshView : public Module {
    MeshIPort* inport;
    GeometryOPort* ogeom;
    int abort_flag;
    
    int numLevels, oldLev;
    int seedTet, oldSeed;
    double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    double clipX, oldClipX, clipY, oldClipY, clipZ, oldClipZ;
    int deep;
    int allLevels;
    int numShare, oldShare;
    Array1<int> levels;

    MaterialHandle mat1;
    MaterialHandle mat2;
public:
    MeshView(const clString& id);
    MeshView(const MeshView&, int deep);
    virtual ~MeshView();
    virtual Module* clone(int deep);
    virtual void execute();
    void initList();
    void addTet(int row, int ind);
    void makeLevels(const MeshHandle&);
};

#endif
