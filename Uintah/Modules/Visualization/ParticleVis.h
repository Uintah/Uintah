#ifndef PARTICLEVIS_H
#define PARTICLEVIS_H

#include <SCICore/TclInterface/TCLvar.h>
#include <Uintah/Datatypes/ScalarParticlesPort.h>
#include <Uintah/Datatypes/VectorParticlesPort.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Uintah {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;

class ParticleVis : public Module {
    ScalarParticlesIPort* spin0;
    ScalarParticlesIPort* spin1;
    VectorParticlesIPort* vpin;
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
       
    int last_idx;
    int last_generation;
    void *cbClass;
 public:
    ParticleVis(const clString& id);
    virtual ~ParticleVis();
    virtual void geom_pick(GeomPick*, void*, int);

    virtual void execute();
};

} // Modules
} // Kurt

#endif
