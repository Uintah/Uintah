/**
 *  \file   ParticlesHelper.h
 *  \date   June, 2014
 *  \author "Tony Saad"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2014 The University of Utah
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

#ifndef Particles_Helper_h
#define Particles_Helper_h

//-- stl includes --//
#include <string>

//-- Wasatch includes --//
#include "GraphHelperTools.h"

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>

//-- Expression includes --//
#include <expression/ExprFwd.h>
#include <expression/ExpressionTree.h>

// forward declarations
namespace Uintah{
  class ProcessorGroup;
  class DataWarehouse;
}

namespace Wasatch {

  class Wasatch;  // forward declaration

  /**
   *  \class  ParticlesHelper
   *  \author Tony Saad
   *  \date   June, 2014
   *  \brief  This class provides support for creating particles.
   */
  class ParticlesHelper
  {
  public:
    
    /**
     * @return the singleton instance of ParticlesHelper
     */
    static ParticlesHelper& self();
    ParticlesHelper();
    ~ParticlesHelper();
    
    void schedule_initialize_particles( const Uintah::LevelP& level,
                                        Uintah::SchedulerP& sched );

    void schedule_sync_particle_position( const Uintah::LevelP& level,
                                  Uintah::SchedulerP& sched,
                                  const bool initialization=false );

    void schedule_transfer_particle_ids( const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched);

    void schedule_relocate_particles( const Uintah::LevelP& level,
                                      Uintah::SchedulerP& sched );
    
    void sync_with_wasatch( Wasatch* const wasatch );

    void schedule_delete_outside_particles( const Uintah::LevelP& level,
                                          Uintah::SchedulerP& sched );

    static void add_particle_variable(const Expr::Tag& varTag);
    
  private:

    const Uintah::VarLabel *pPosLabel_, *pIDLabel_;
    const Uintah::VarLabel *pXLabel_,*pYLabel_,*pZLabel_;
    std::vector<const Uintah::VarLabel*> destroyMe_;
    
    static std::vector<Expr::Tag> otherParticleTags_;
    
    void initialize_particles(const Uintah::ProcessorGroup*,
                         const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                         Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    void transfer_particle_ids(const Uintah::ProcessorGroup*,
                              const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                              Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    void delete_outside_particles(const Uintah::ProcessorGroup*,
                              const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                              Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    void sync_particle_position(const Uintah::ProcessorGroup*,
                        const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                        Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw, const bool initialization);

    void relocate_particles(const Uintah::ProcessorGroup*,
                        const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                        Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    Wasatch* wasatch_;
    bool wasatchSync_;
    int nParticles_; // number of initial particles
    Uintah::ProblemSpecP particleEqsSpec_;
  };

} /* namespace Wasatch */
#endif /* Reduction_Helper_h */