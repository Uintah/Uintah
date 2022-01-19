/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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


#ifndef Uintah_Models_ParticleBased_TracerParticles_h
#define Uintah_Models_ParticleBased_TracerParticles_h

#include <CCA/Components/Models/ParticleBased/ParticleModel.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <map>
#include <vector>

namespace Uintah {

  class ICELabel;
  class TracerParticles : public ParticleModel {
  public:
    TracerParticles(const ProcessorGroup    * myworld,
                    const MaterialManagerP  & materialManager,
                    const ProblemSpecP      & params);

    virtual ~TracerParticles();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid,
                               const bool isRestart);

    virtual void scheduleInitialize(SchedulerP   &,
                                    const LevelP & level);

    virtual void restartInitialize() {};

    virtual void scheduleComputeModelSources(SchedulerP   &,
                                             const LevelP & level);
   

  //______________________________________________________________________
  //        
  private:
    
    //__________________________________
    // labels
    ICELabel* Ilb;                                      
    VarLabel * pXLabel;           // particle position label
    VarLabel * pXLabel_preReloc;
    VarLabel * pDispLabel;
    VarLabel * pDispLabel_preReloc;
    VarLabel * pIDLabel;          // particle ID label, of type long64
    VarLabel * pIDLabel_preReloc;
    
    VarLabel * nPPCLabel;         // number of particles in a cell

    //__________________________________
    //  Region used for initialization
    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);

      GeometryPieceP piece;
      IntVector ppc;          // particles per cell
    };

    //__________________________________
    //  For injecting a scalar inside the domain
    class interiorRegion {
    public:
      interiorRegion(GeometryPieceP piece, ProblemSpecP&);
      GeometryPieceP piece;
      IntVector ppc;          // particles per cell
    };

    //__________________________________
    //
    class Tracer {
    public:
      std::string name;
      std::string fullName;

      std::vector<Region*> regions;
      std::vector<interiorRegion*> interiorRegions;
    };

    Tracer* d_tracer;
    
    
    //__________________________________
    //  typedefs to help laying down particles
    typedef std::map<Region*, std::vector<Point> > regionPoints;
    
    //__________________________________
    //
    unsigned int countParticles( const Patch   * patch,
                                 regionPoints  & pPositions);

    void initialize(const ProcessorGroup  *,
                    const PatchSubset     * patches,
                    const MaterialSubset  * matls,
                    DataWarehouse         *,
                    DataWarehouse         * new_dw);
                             
    void sched_updateParticles(SchedulerP  & sched,
                               const LevelP& level);

    void updateParticles(const ProcessorGroup  *,
                         const PatchSubset     * patches,                  
                         const MaterialSubset  * matls,                    
                         DataWarehouse         * old_dw,                   
                         DataWarehouse         * new_dw);                  

    TracerParticles(const TracerParticles&);
    TracerParticles& operator=(const TracerParticles&);


    //__________________________________
    //
    ProblemSpecP    d_params;
    Ghost::GhostType  d_gn  = Ghost::None;
    Ghost::GhostType  d_gac = Ghost::AroundCells;

  };
}

#endif
