#ifndef PARTICLEVIS_H
#define PARTICLEVIS_H

#include <Packages/Uintah/Core/Datatypes/ScalarParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/TclInterface/TCLvar.h>
#include <Dataflow/Network/Module.h>

namespace Uintah {
using namespace SCIRun;

class ParticleVis : public Module {
    ScalarParticlesIPort* spin0;
    ScalarParticlesIPort* spin1;
    VectorParticlesIPort* vpin;
    TensorParticlesIPort* tpin;
    ColorMapIPort *cin;
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

    bool hasIDs;    int last_idx;
    int last_generation;
    void *cbClass;
 public:
    ParticleVis(const clString& id);
    virtual ~ParticleVis();
    virtual void geom_pick(GeomPick*, void*, GeomObj*);
    virtual void execute();
};
} // End namespace Uintah


#endif
