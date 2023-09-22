/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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
#include <Core/Grid/Variables/VarLabel.h>

#include <map>
#include <vector>
#include <memory>

namespace Uintah {
  class VarLabel;
  class ICELabel;
  class Patch;


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

    virtual void scheduleRestartInitialize(SchedulerP&,
                                           const LevelP& level);

    virtual void scheduleComputeModelSources(SchedulerP   &,
                                             const LevelP & level);


  //______________________________________________________________________
  //
  private:

    // remove this once C++14 is adopted
    template<typename T, typename ...Args>
    std::unique_ptr<T> make_unique( Args&& ...args );
    enum modifiesComputes{ modifiesVar, computesVar};

    //__________________________________
    // labels
    ICELabel* Ilb;

    VarLabel * pDispLabel;
    VarLabel * pDispLabel_preReloc;
    VarLabel * pVelocityLabel;
    VarLabel * pVelocityLabel_preReloc;
    VarLabel * nPPCLabel;         // number of particles in a cell
    VarLabel * simTimeLabel;

    //__________________________________
    //  Variables that will be used to set tracer particle values
    struct cloneVar{

      cloneVar(){};

      int matl;
      std::string CCVarName;
      VarLabel *  CCVarLabel      {nullptr};
      VarLabel *  pQLabel_preReloc{nullptr};
      VarLabel *  pQLabel         {nullptr};

      ~cloneVar()
      {
        VarLabel::destroy( pQLabel_preReloc );
        VarLabel::destroy( pQLabel );
      }
    };

    //__________________________________
    //  tracer value will be set via a decay model
    //  This is used in conjunction with the passiveScalar model
    struct scalar{

      scalar(){};

      // for exponential decay model
      enum decayCoef{ constant, variable, none};
      decayCoef  decayCoefType = none;

      bool withExpDecayModel {false};
      double  c1 {-9};
      double  c2 {-9};
      double  c3 {-9};
      std::string c2_filename {"-9"};

      int matl;
      double  initialValue {-9};

      std::string labelName;
      VarLabel *  expDecayCoefLabel       {nullptr};
      VarLabel *  totalDecayLabel_preReloc{nullptr};
      VarLabel *  totalDecayLabel         {nullptr};
      VarLabel *  label_preReloc {nullptr};
      VarLabel *  label          {nullptr};

      // container for preReloc label and initial value
      std::multimap<VarLabel*, double> label_value;

      ~scalar()
      {
        VarLabel::destroy( totalDecayLabel_preReloc);
        VarLabel::destroy( totalDecayLabel);
        VarLabel::destroy( label_preReloc );
        VarLabel::destroy( label );
        VarLabel::destroy( expDecayCoefLabel );
      }
    };

    //__________________________________
    //  Region used for initialization
    //  and adding particles
    //  This is NOT a geom_object but a geom_piece
    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);

      GeometryPieceP piece;
      int particlesPerCell          {8};     // particles per cell
      int particlesPerCellPerSecond {0};     // particles per cell per second

      bool isInteriorRegion         {false};          // flag for particle injection
      std::map<const Patch* ,double> elapsedTime;     //  Elapsed time since particles were added
    };

    //__________________________________
    //
    class Tracer {
    public:
      std::string name;
      std::string fullName;

      std::vector<Region*> initializeRegions;   // regions where particles are initialized
      std::vector<Region*> injectionRegions;    // regions where particles are injected

      double timeStart;
      double timeStop;
    };

    Tracer* d_tracer;


    //__________________________________
    //  typedefs to help laying down particles
    typedef std::map<Region*, std::vector<Point> > regionPoints;

    //__________________________________
    //
    unsigned int distributeParticles( const Patch   * patch,
                                      const double    simTime,
                                      const double    delT,
                                      const std::vector<Region*> regions,
                                      regionPoints  & pPositions);

    void initializeCoreVariables( const Patch   * patch,
                                  unsigned int    pIndx,
                                  regionPoints  & pPositions,
                                  std::vector<Region*> regions,
                                  ParticleVariable<Point> & pX,
                                  ParticleVariable<Vector>& pDisp,
                                  ParticleVariable<Vector>& pVelocity,
                                  ParticleVariable<long64>& pID,
                                  CCVariable<int>         & nPPC );

    void initializeCloneVars( ParticleSubset * pset,
                              const Patch    * patch,
                              const int        indx,
                              DataWarehouse  * new_dw );
                              
    void initializeScalarVars( ParticleSubset * pset,
                               const Patch    * patch,
                               const int        indx,
                               DataWarehouse  * new_dw,
                               const modifiesComputes which);

    void initializeRegions( const Patch             *  patch,
                            unsigned int               pIndx,
                            regionPoints             & pPositions,
                            std::vector<Region*>       regions,
                            constCCVariable<double>  & Q_CC,
                            const double               initialValue,
                            ParticleVariable<double> & pQ  );



    void initializeTask(const ProcessorGroup  *,
                        const PatchSubset     * patches,
                        const MaterialSubset  * matls,
                        DataWarehouse         *,
                        DataWarehouse         * new_dw);

    void restartInitializeTask(const ProcessorGroup  *,
                               const PatchSubset     * patches,
                               const MaterialSubset  * matls,
                               DataWarehouse         *,
                               DataWarehouse         * new_dw);

    void sched_restartInitializeHACK( SchedulerP&,
                                      const LevelP& level);

    void restartInitializeHACK( const ProcessorGroup  *,
                                const PatchSubset     * patches,
                                const MaterialSubset  * matls,
                                DataWarehouse         *,
                                DataWarehouse         * new_dw){};

    void sched_moveParticles(SchedulerP  & sched,
                             const LevelP& level);

    void moveParticles(const ProcessorGroup  *,
                       const PatchSubset     * patches,
                       const MaterialSubset  * matls,
                       DataWarehouse         * old_dw,
                       DataWarehouse         * new_dw);

    void sched_addParticles( SchedulerP  & sched,
                             const LevelP& level);

    void addParticles(const ProcessorGroup  *,
                      const PatchSubset     * patches,
                      const MaterialSubset  * ,
                      DataWarehouse         * old_dw,
                      DataWarehouse         * new_dw);

    void sched_setParticleVars( SchedulerP  & sched,
                                const LevelP& level);

    void setParticleVars(const ProcessorGroup  *,
                         const PatchSubset     * patches,
                         const MaterialSubset  * ,
                         DataWarehouse         * ,
                         DataWarehouse         * new_dw );

    TracerParticles(const TracerParticles&);
    TracerParticles& operator=(const TracerParticles&);


    //__________________________________
    //
    ProblemSpecP    d_params;
    Ghost::GhostType  d_gn  = Ghost::None;
    Ghost::GhostType  d_gac = Ghost::AroundCells;
    std::vector< std::shared_ptr< cloneVar >>    d_cloneVars;
    std::vector< std::shared_ptr< scalar >>      d_scalars;
    
    
    bool d_previouslyInitialized {false};            // this is set in a checkpoint and checked in ProblemSetup
    bool d_reinitializeDomain    {false};            // to erase the previous particles and start over
  };
}

#endif
