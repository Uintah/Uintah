/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef PARTICLEVIS_H
#define PARTICLEVIS_H

#include <Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Uintah/Dataflow/Ports/VectorParticlesPort.h>
#include <Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>

namespace Uintah {
using namespace SCIRun;

class ParticleVis : public Module {
    ScalarParticlesIPort* spin0;
    ScalarParticlesIPort* spin1;
    VectorParticlesIPort* vpin;
    TensorParticlesIPort* tpin;
    MatrixIPort *mat_in;
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
