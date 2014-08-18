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

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>

namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;

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
    virtual ~ParticlesHelper();

    /*
     *  \brief Initializes particle memory. This must be called at schedule initialize. DO NOT call
     it on restarts. Use schedule_restart_initialize instead
     */
    virtual void schedule_initialize ( const Uintah::LevelP& level,
                                       Uintah::SchedulerP& sched   );

    /*
     *  \brief Will reallocate some essential objects for the particles helper such as the delete sets.
     MUST be called at restarts. Since there is no way to schedule restart initialize, you should call this
     function somewhere in scheduleTimeAdvance. See the Wasatch example.
     */
    virtual void schedule_restart_initialize( const Uintah::LevelP& level,
                                              Uintah::SchedulerP& sched   );

    /*
     *  \brief Task that synchronizes Wasatch-specific particle positions (x, y, w) that are store
     as sepearte fields with the Uintah::Point position field. The Uintah::Point position field 
     is essential for the relocation algorithm to Work. Wasatch does not support particle fields
     of type Point. You should call this at the end of the scheduleTimeAdvance, after all internal
     Wasatch tasks have completed.
     */
    virtual void schedule_sync_particle_position( const Uintah::LevelP& level,
                                                  Uintah::SchedulerP& sched,
                                                  const bool initialization=false );

    /*
     *  \brief Task that transfers particle IDs from the oldDW to the new DW.
     */
    virtual void schedule_transfer_particle_ids( const Uintah::LevelP& level,
                                                 Uintah::SchedulerP& sched);

    /*
     *  \brief Task that schedules the relocation algorithm. Should be the last function called
     on particles in the scheduleTimeAdvance.
     */
    virtual void schedule_relocate_particles( const Uintah::LevelP& level,
                                              Uintah::SchedulerP& sched );

    virtual void schedule_delete_outside_particles( const Uintah::LevelP& level,
                                                    Uintah::SchedulerP& sched );

    static void add_particle_variable( const std::string& varName );

    /*
     *  \brief Parse the particle spec and create the position varlabels. This is an essential step
     that MUST be called during Wasatch::ProblemSetup
     */
    void problem_setup(Uintah::ProblemSpecP particleEqsSpec);
    void set_materials(const Uintah::MaterialSet* const materials)
    {
      isValidState_ = true;
      materials_ = materials;
    }
    
  protected:

    const Uintah::VarLabel *pPosLabel_, *pIDLabel_;
    const Uintah::VarLabel *pXLabel_,*pYLabel_,*pZLabel_;
    std::vector<const Uintah::VarLabel*> destroyMe_;
    
    static std::vector<std::string> otherParticleVarNames_;
    
    virtual void initialize( const Uintah::ProcessorGroup*,
                               const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                               Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void restart_initialize( const Uintah::ProcessorGroup*,
                             const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                             Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void transfer_particle_ids(const Uintah::ProcessorGroup*,
                              const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                              Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void delete_outside_particles(const Uintah::ProcessorGroup*,
                              const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                              Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void clear_deleteset(const Uintah::ProcessorGroup*,
                          const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                          Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void sync_particle_position(const Uintah::ProcessorGroup*,
                        const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                        Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw, const bool initialization);

    bool isValidState_;
    const Uintah::MaterialSet* materials_;
    int nParticles_; // number of initial particles
    Uintah::ProblemSpecP particleEqsSpec_;
    std::map<int, Uintah::ParticleSubset*> deleteSet_;
  };

} /* namespace Wasatch */
#endif /* Reduction_Helper_h */