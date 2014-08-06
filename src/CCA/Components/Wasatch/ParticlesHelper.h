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

/**************************************************************************************
 /       \  /      \ /       \ /        |/      \ /      |/  |      /        | /      \
 $$$$$$$  |/$$$$$$  |$$$$$$$  |$$$$$$$$//$$$$$$  |$$$$$$/ $$ |      $$$$$$$$/ /$$$$$$  |
 $$ |__$$ |$$ |__$$ |$$ |__$$ |   $$ |  $$ |  $$/   $$ |  $$ |      $$ |__    $$ \__$$/
 $$    $$/ $$    $$ |$$    $$<    $$ |  $$ |        $$ |  $$ |      $$    |   $$      \
 $$$$$$$/  $$$$$$$$ |$$$$$$$  |   $$ |  $$ |   __   $$ |  $$ |      $$$$$/     $$$$$$  |
 $$ |      $$ |  $$ |$$ |  $$ |   $$ |  $$ \__/  | _$$ |_ $$ |_____ $$ |_____ /  \__$$ |
 $$ |      $$ |  $$ |$$ |  $$ |   $$ |  $$    $$/ / $$   |$$       |$$       |$$    $$/
 $$/       $$/   $$/ $$/   $$/    $$/    $$$$$$/  $$$$$$/ $$$$$$$$/ $$$$$$$$/  $$$$$$/
**************************************************************************************/

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
  
  //==================================================================
  /**
   *  \class  ParticlesHelper
   *  \author Tony Saad
   *  \date   June, 2014
   *  \brief  This class provides support for creating particles.
   */
  class ParticlesHelper
  {
  public:
    
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

    /*
     *  \brief Task that deletes particles outside the bounds of a patch. This is usually taken care
     of by the relocation alogrithm which moves these particles to neighboring patches. However, since
     the relocation algorithm does not work at initialization, this task can be used instead. Sometimes
     when creating particles in a patch, some particles are created outside the patch bounds. This causes
     problems with particle operators. Use this task to cleanup those outside particles.
     */
    virtual void schedule_delete_outside_particles( const Uintah::LevelP& level,
                                                    Uintah::SchedulerP& sched );

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
    
    /**
     * \brief Use to add particle variable names marked for relocation. Note, do NOT add particle
     position to this list.
     */
    static void add_particle_variable(const std::string& varName);
    
    /**
     * \brief Return a reference to the list of particle variables marked for relocation.
     */
    static const std::vector<std::string>& get_relocatable_particle_varnames();

    /**
     * \brief Return a reference to the list of particle variables marked for relocation.
     */
    static const std::vector<int>* get_boundary_particles(const std::string& bndName,
                                                          const int patchID);
    /**
     * \brief Schedule this task whenever you want to update the list of particles that are near boundaries.
     */
    virtual void schedule_find_boundary_particles(const Uintah::LevelP& level,
                                                  Uintah::SchedulerP& sched);
  protected:
    
    
    //****************************************************************************
    /**
     *  @struct ParticleBoundarySpec
     *  @author Tony Saad
     *  @date   July 2014
     *  @brief  Stores boundary information in a convenient way.
     */
    //****************************************************************************
    struct ParticleBoundarySpec
    {
      std::string              name;      // name of the boundary
      Uintah::Patch::FaceType  face;      // x-minus, x-plus, y-minus, y-plus, z-minus, z-plus
      std::vector<int>         patchIDs;  // List of patch IDs that this boundary lives on.
      // returns true if this Boundary has parts of it on patchID
      bool has_patch(const int& patchID) const
      {
        return std::find(patchIDs.begin(), patchIDs.end(), patchID) != patchIDs.end();
      };
    };
    
    // This vector stores a list of all particle variables that require relocation(except position).
    static std::vector<std::string> otherParticleVarNames_;

    typedef std::vector<int> BndParticlesVector;
    typedef std::map <int, BndParticlesVector            > patchIDBndParticlesMapT;  // temporary typedef map that stores boundary particles per patch id: Patch ID -> Bnd particles
    typedef std::map <std::string, patchIDBndParticlesMapT    > MaskMapT         ;  // boundary name -> (patch ID -> boundary particles )
    static MaskMapT bndParticlesMap_;

    // particle position label of type Uintah::Point
    const Uintah::VarLabel *pPosLabel_;
    // particle ID label, of type long64
    const Uintah::VarLabel *pIDLabel_;
    // particle x, y, and z position (of type double)
    const Uintah::VarLabel *pXLabel_,*pYLabel_,*pZLabel_;
    
    // list of varlabels to be destroyed
    std::vector<const Uintah::VarLabel*> destroyMe_;
    
    virtual void initialize( const Uintah::ProcessorGroup*,
                             const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                             Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw );

    virtual void restart_initialize( const Uintah::ProcessorGroup*,
                                    const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                    Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw );

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

    void allocate_boundary_particles_vector( const std::string& bndName,
                                             const int& patchID );

    void update_boundary_particles_vector( const std::vector<int>& myIters,
                                           const std::string& bndName,
                                           const int& patchID );
    
    void find_boundary_particles( const Uintah::ProcessorGroup*,
                                  const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                 Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw );

    /**
     * \brief Use this to parse boundary conditions specified in the input file and allocate appropriate
     memory for the boundary particles.
     */
    virtual void parse_boundary_conditions(const Uintah::PatchSet* const patches);
    
    /**
     * \brief Use this to parse boundary conditions specified in the input file and allocate appropriate
     memory for the boundary particles. This function will call parse_boundary_conditions(const Uintah::PatchSet* const patches)
     */
    virtual void parse_boundary_conditions(const Uintah::LevelP& level,
                                           Uintah::SchedulerP& sched);

    bool isValidState_;
    const Uintah::MaterialSet* materials_;
    double pPerCell_; // number of initial particles per cell
    unsigned int maxParticles_;
    Uintah::ProblemSpecP particleEqsSpec_;
    std::map<int, Uintah::ParticleSubset*> deleteSet_;
  }; // Class ParticlesHelper

} /* namespace Uintah */

#endif /* Particles_Helper_h */