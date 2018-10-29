/**
 *  \file   ParticlesHelper.h
 *  \date   June, 2014
 *  \author "Tony Saad"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2018 The University of Utah
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
#include <Core/Grid/MaterialManager.h>

#ifndef PIDOFFSET
#define PIDOFFSET 1000000000000ul
#endif

namespace Uintah {

  class ProcessorGroup;
  class DataWarehouse;
  
  //==================================================================
  /**
   *  \class    ParticlesHelper
   
   *  \author   Tony Saad
   
   *  \date     June, 2014
   
   *  \brief    This class provides support for creating and managing particles in Uintah.

   *  \details  Uintah provides support for tracking particles using a ParticleSubset and a ParticleVariable.
   A ParticleSubset represents the memory size of the particle field (e.g. number of particles on a
   given patch and material). A ParticleVariable, as the name designates, represents a unique variable
   associated with a particle. The size of a ParticleVariable is based on the ParticleSubset. As a 
   rule of thumb, one will usually have one ParticleSubset per patch, per material, and an arbitrary
   number of ParticleVariable vars per ParticleSubset.

   A Uintah ParticleVariable can be templated on a variety of types (e.g. int, double, Point)
   and is typically an array whose values are associated with different particles. An essential
   variable that is intimately built into the Uintah particle support framework is the particle
   position vector. This is a ParticleVariable templated on Uintah::Point, in other words, it is
   an array of triplets. The particle position is required for load balancing, particle relocation
   across patches, and visualization. From here on, ParticleVariable<Point> is referred to as the
   particle position vector.
   
   Unfortunately, there are several reasons that require one to track particle position as three
   separate particle variables instead of using a particle position vector. These include design
   and personal preferences, and most importantly, the use of the Nebo EDSL. The Nebo EDSL was designed
   (and for good reasons) to deal with vectors of doubles. It cannot handle vectors of triplets.
   That being said, the ParticlesHelper is designed with Nebo in mind and will therefore require
   that you create three particle variables of type double for the three particle positions.   
   
   *
   *  \note: Visit currently makes the assumption that the particle position vector be named "p.x".
   This will change in the next Uintah release.
   *  \note: Particle variable names typically start with "p." - a Uintah assumption made to simplify
   some runtime decisions.
   *  \note: You will need to provide a basic xml support for particles:
   \code{.xml}
   <ParticlesPerCell spec="OPTIONAL DOUBLE"/>
   <MaximumParticles spec="OPTIONAL INTEGER"/>
   <ParticlePosition spec="REQUIRED NO_DATA"
                     attribute1="x REQUIRED STRING"
                     attribute2="y REQUIRED STRING"
                     attribute3="z REQUIRED STRING"/>
   \endcode
   */
  class ParticlesHelper
  {
  public:
    
    ParticlesHelper();
    virtual ~ParticlesHelper();

    /*
     *  \brief Creates particle memory (ParticleSubset) which can be used to allocated particle variables. 
     Also, this function will allocate memory for pPosLabel_ and pIDLabel_ without initializing them
     given the assumptions made for this class. The component is supposed to track particle position
     for x, y, and z seperately and then synchronize with the particle position vector required by 
     Uintah. Finally, this function must be called in a component's schedule_initialize.
     *
     *  \note: This function will NOT allocate memory for pXLabel_, pYLabel_, and PZLabel_. You are 
     responsible for allocating and initializing those.
     *
     *  \warning DO NOT call this function on restarts. Use schedule_restart_initialize instead for restarts.
     */
    virtual void schedule_initialize ( const Uintah::LevelP& level,
                                       Uintah::SchedulerP& sched   );

    /*
     *  \brief Will reallocate some essential objects for the particles helper such as the delete sets.
     *
     *  \warning MUST be called at restarts. Since there is no way to schedule restart initialize, 
     you should call this function somewhere in scheduleTimeAdvance. See how Wasatch handles this.
     */
    virtual void schedule_restart_initialize( const Uintah::LevelP& level,
                                              Uintah::SchedulerP& sched   );

    /*
     *  \brief Task that synchronizes particle positions (x, y, w) that are stored
     as sepearte fields with the Uintah::Point particle position vector. The Uintah::Point particle
     position vector is essential for the relocation algorithm to Work. You should call this at the end of the
     scheduleTimeAdvance, after all your particle transport has completed.
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
     that MUST be called during Component::ProblemSetup
     */
    void problem_setup(Uintah::ProblemSpecP uintahSpec,
                       Uintah::ProblemSpecP particleEqsSpec);
    
    void set_materials(const Uintah::MaterialSet* const materials)
    {
      isValidState_ = true;
      materials_ = materials;
    }
    
    /**
     * \brief Use to mark a particle variable for relocation. Please do NOT add the Uintah particle
     position vector ("p.x") to this list.
     */
    static void mark_for_relocation(const std::string& varName);

    /**
     * \brief Use to mark which particle variables require a boundary condition. Usually, these
     are the transported particle variables. This is needed by the particle addition algorithm
     */
    static void needs_boundary_condition(const std::string& varName);
    
    /**
     * \brief Return a reference to the list of particle variables marked for relocation.
     */
    static const std::vector<std::string>& get_relocatable_particle_varnames();

    /**
     * \brief Task that adds particles through boundaries. This task will parse the ups input file and
     add particles through the boundaries as specified.
     * \warning  Use AFTER schedule_particle_relocation.
     */
    virtual void schedule_add_particles( const Uintah::LevelP& level,
                                         Uintah::SchedulerP& sched );
    
    /**
     * \brief Return a reference to the list of particle near the boundary specified by bndName and on patch ID 
     specified via patchID.
     */
    static const std::vector<int>* get_boundary_particles(const std::string& bndName,
                                                          const int patchID);
    /**
     * \brief Schedule this task whenever you want to update the list of particles that are near boundaries.
     */
    virtual void schedule_find_boundary_particles(const Uintah::LevelP& level,
                                                  Uintah::SchedulerP& sched);
  protected:
        
    // This vector stores a list of all particle variables that require relocation(except position).
    static std::vector<std::string> needsRelocation_;
    static std::vector<std::string> needsBC_;

    typedef std::vector<int> BndParticlesVector;
    typedef std::map <int, BndParticlesVector> patchIDBndParticlesMapT; // temporary typedef map that stores boundary particles per patch id: Patch ID -> Bnd particles
    typedef std::map <std::string, patchIDBndParticlesMapT> MaskMapT;   // boundary name -> (patch ID -> boundary particles )
    static MaskMapT bndParticlesMap_;
    static MaskMapT inletBndParticlesMap_;

    // particle position label of type Uintah::Point
    const Uintah::VarLabel *pPosLabel_;
    // particle ID label, of type long64
    const Uintah::VarLabel *pIDLabel_;
    // particle x, y, and z position (of type double)
    const Uintah::VarLabel *pXLabel_,*pYLabel_,*pZLabel_;

    const Uintah::VarLabel *delTLabel_;

    static std::string pPosName_, pIDName_;
    
    // list of varlabels to be destroyed
    std::vector<const Uintah::VarLabel*> destroyMe_;
    
    virtual void initialize( const Uintah::ProcessorGroup*,
                             const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                             Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw );

    virtual void restart_initialize( const Uintah::ProcessorGroup*,
                                     const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                     Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw );

    virtual void transfer_particle_ids( const Uintah::ProcessorGroup*,
                                        const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                        Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void delete_outside_particles( const Uintah::ProcessorGroup*,
                                           const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                           Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void clear_deleteset( const Uintah::ProcessorGroup*,
                                  const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                  Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void sync_particle_position( const Uintah::ProcessorGroup*,
                                         const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                         Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw, const bool initialization);
    virtual void sync_particle_position_periodic( const Uintah::ProcessorGroup*,
                                        const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                        Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    virtual void add_particles( const Uintah::ProcessorGroup*,
                                const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw );

    void allocate_boundary_particles_vector( const std::string& bndName,
                                             const int& patchID );

    void update_boundary_particles_vector( const std::vector<int>& myIters,
                                           const std::string& bndName,
                                           const int& patchID );
    
    void find_boundary_particles( const Uintah::ProcessorGroup*,
                                  const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                                  Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw );

    void initialize_internal(const int materialSize);
    
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
    unsigned int maxParticles_; //number of maximum initial particles
    Uintah::ProblemSpecP particleEqsSpec_;
    std::vector< std::map<int, Uintah::ParticleSubset*> > deleteSets_; // material [ patchID -> last particle ID ]
    std::vector< std::map<int, long64> > lastPIDPerMaterialPerPatch_;  // material [ patchID -> last particle ID ]
  }; // Class ParticlesHelper

} /* namespace Uintah */

#endif /* Particles_Helper_h */
