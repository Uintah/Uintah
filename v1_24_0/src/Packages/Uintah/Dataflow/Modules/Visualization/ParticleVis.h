#ifndef PARTICLEVIS_H
#define PARTICLEVIS_H

#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Packages/Uintah/Dataflow/Ports/VectorParticlesPort.h>
#include <Packages/Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/GuiInterface/GuiVar.h>
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
    GuiDouble min_;
    GuiDouble max_;
    GuiInt isFixed;
    GuiDouble current_time;
    GuiDouble radius;
    GuiInt auto_radius;
    GuiInt drawcylinders;
    GuiDouble length_scale;
    GuiInt auto_length_scale;
    GuiDouble min_crop_length;
    GuiDouble max_crop_length;
    GuiDouble head_length;
    GuiDouble width_scale;
    GuiDouble shaft_rad;
    GuiInt show_nth;
    GuiInt drawVectors;
    GuiInt drawspheres;
    GuiInt polygons; // number of polygons used to represent
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
  ParticleVis(GuiContext* ctx);
    virtual ~ParticleVis();
    virtual void geom_pick(GeomPickHandle, void*, GeomHandle);
    virtual void execute();
};
} // End namespace Uintah


#endif
