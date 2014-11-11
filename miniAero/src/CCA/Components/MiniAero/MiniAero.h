/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CCA_COMPONENTS_MINIAERO_H
#define CCA_COMPONENTS_MINIAERO_H

#include <CCA/Ports/SimulationInterface.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>

namespace Uintah {

class SimpleMaterial;

/**************************************

 CLASS
 MiniAero

 MiniAero simulation


 GENERAL INFORMATION

 MiniAero.h


 KEYWORDS
 MiniAero


 DESCRIPTION
 2D implementation of MiniAero's Algorithm using Euler's method.

 ****************************************/

class MiniAero : public UintahParallelComponent, public SimulationInterface {

  public:

    MiniAero(const ProcessorGroup* myworld);

    virtual ~MiniAero();

    virtual void problemSetup(const ProblemSpecP& params,
                              const ProblemSpecP& restart_prob_spec,
                              GridP& grid,
                              SimulationStateP&);

    virtual void scheduleInitialize(const LevelP& level,
                                    SchedulerP& sched);

    virtual void scheduleComputeStableTimestep(const LevelP& level,
                                               SchedulerP&);

    virtual void scheduleTimeAdvance(const LevelP& level,
                                     SchedulerP&);

  private:

    void getGeometryObjects(ProblemSpecP& ps, 
                            std::vector<GeometryObject*>& geom_objs);

    void initializeCells(CCVariable<double>& rho_CC,
                         CCVariable<double>& temp_CC,
                         CCVariable<Vector>& vel_CC,
                         const Patch* patch,
                         DataWarehouse* new_dw,
                         std::vector<GeometryObject*>& geom_objs);

    void initialize(const ProcessorGroup*,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw);

    void computeStableTimestep(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw);


    void schedConvertOutput(const LevelP& level,SchedulerP& sched);

    void schedCellCenteredFlux(const LevelP& level,SchedulerP& sched);
    void schedFaceCenteredFlux(const LevelP& level,SchedulerP& sched);
    void schedUpdateResidual(const LevelP& level,SchedulerP& sched);
    void schedUpdateState(const LevelP& level,SchedulerP& sched);
    
    void convertOutput(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);

    void cellCenteredFlux(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);

    void faceCenteredFlux(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);
    void updateResidual(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);
    void updateState(const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw);

    const VarLabel* conserved_label;
    const VarLabel* rho_CClabel;
    const VarLabel* vel_CClabel;
    const VarLabel* press_CClabel;
    const VarLabel* temp_CClabel;
    const VarLabel* viscosityLabel;
    const VarLabel* speedSound_CClabel;
    const VarLabel* flux_mass_CClabel;
    const VarLabel* flux_mom_CClabel;
    const VarLabel* flux_energy_CClabel;
    const VarLabel* flux_mass_FCXlabel;
    const VarLabel* flux_mom_FCXlabel;
    const VarLabel* flux_energy_FCXlabel;
    const VarLabel* flux_mass_FCYlabel;
    const VarLabel* flux_mom_FCYlabel;
    const VarLabel* flux_energy_FCYlabel;
    const VarLabel* flux_mass_FCZlabel;
    const VarLabel* flux_mom_FCZlabel;
    const VarLabel* flux_energy_FCZlabel;
    const VarLabel* residual_CClabel;

    SimulationStateP sharedState_;
    double delt_;
    double d_gamma;
    double d_R;
    double d_CFL;
    bool d_viscousFlow;
    SimpleMaterial* mymat_;
    std::vector<GeometryObject*> d_geom_objs;

    MiniAero(const MiniAero&);
    MiniAero& operator=(const MiniAero&);

    double getViscosity(const double & temp) const
    {
      return 1.458E-6*pow(temp,1.5)/(temp+110.4);
    }

    double getGamma() const
    {
      return d_gamma;
    }


};  // end class MiniAero

}  // end namespace Uintah

#endif  // end CCA_COMPONENTS_MINIAERO_H
