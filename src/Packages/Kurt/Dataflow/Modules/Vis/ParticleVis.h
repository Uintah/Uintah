#ifndef PARTICLE_VIS_H
#define PARTICLE_VIS_H

#include <Core/TclInterface/TCLvar.h>
#include <Packages/Kurt/DataArchive/VisParticleSetPort.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Kurt {
using namespace SCIRun;

class ParticleVis : public Module {
    VisParticleSetIPort* iPort;
    ColorMapIPort *iCmap;
    GeometryOPort* ogeom;
    TCLdouble current_time;
    TCLdouble radius;
    TCLint drawcylinders;
    TCLdouble length_scale;
    TCLdouble head_length;
    TCLdouble width_scale;
    TCLdouble shaft_rad;
    TCLint show_nth;
    TCLint drawVectors;
    TCLint drawspheres;
    TCLint polygons; // number of polygons used to represent
    // a sphere: [minPolys, MAX_POLYS]
    const int MIN_POLYS;    // polys, nu, and nv must correlate
    const int MAX_POLYS;  // MIN_NU*MIN_NV = MIN_POLYS
    const int MIN_NU;       // MAX_NU*MAX_NV = MAX_POLYS
    const int MAX_NU;
    const int MIN_NV;
    const int MAX_NV;
    MaterialHandle outcolor;
       
    int last_idx;
    int last_generation;
    void *cbClass;
 public:
    ParticleVis(const clString& id);
    virtual ~ParticleVis();
    virtual void geom_pick(GeomPick*, void*, GeomObj*);

    virtual void execute();
};
} // End namespace Kurt


#endif

