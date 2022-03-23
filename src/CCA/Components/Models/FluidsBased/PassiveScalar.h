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


#ifndef Packages_Uintah_CCA_Components_Examples_PassiveScalar_h
#define Packages_Uintah_CCA_Components_Examples_PassiveScalar_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <map>
#include <vector>

namespace Uintah {

/**************************************

CLASS
   PassiveScalar

   PassiveScalar simulation

GENERAL INFORMATION

   PassiveScalar.h

   Todd Harman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   PassiveScalar

DESCRIPTION
   Long description...

WARNING

****************************************/
  class ICELabel;
  class PassiveScalar :public FluidsBasedModel {
  public:
    PassiveScalar(const ProcessorGroup* myworld,
                  const MaterialManagerP& materialManager,
                  const ProblemSpecP& params);

    virtual ~PassiveScalar();

    virtual void outputProblemSpec(ProblemSpecP& ps);

    virtual void problemSetup(GridP& grid,
                               const bool isRestart);

    virtual void scheduleInitialize(SchedulerP&,
                                    const LevelP& level);

    virtual void scheduleRestartInitialize(SchedulerP&,
                                           const LevelP& level);

    virtual void scheduleComputeStableTimeStep(SchedulerP&,
                                               const LevelP& level);

    virtual void scheduleComputeModelSources(SchedulerP&,
                                             const LevelP& level);

   virtual void scheduleModifyThermoTransportProperties(SchedulerP&,
                                                        const LevelP&,
                                                        const MaterialSet*);

   virtual void computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,
                                    DataWarehouse*,
                                    const int);

   virtual void scheduleErrorEstimate(const LevelP& coarseLevel,
                                      SchedulerP& sched);

   virtual void scheduleTestConservation(SchedulerP&,
                                         const PatchSet* patches);
   
   // create a static function so TracerParticles can call it. 
   static void readTable( const Patch * patch,
                          const Level * level,
                          const std::string filename,
                          CCVariable<double>& c2 );

  private:
    ICELabel* Ilb;

   void modifyThermoTransportProperties(const ProcessorGroup  *,
                                        const PatchSubset     * patches,
                                        const MaterialSubset  *,
                                        DataWarehouse         *,
                                        DataWarehouse         * new_dw);

    void initialize(const ProcessorGroup  *,
                    const PatchSubset     * patches,
                    const MaterialSubset  * matls,
                    DataWarehouse         *,
                    DataWarehouse         * new_dw);

    void  restartInitialize(const ProcessorGroup *,
                            const PatchSubset    * patches,
                            const MaterialSubset *,
                            DataWarehouse        * ,
                            DataWarehouse        * new_dw);

    void computeModelSources(const ProcessorGroup *,
                             const PatchSubset    * patches,
                             const MaterialSubset *,
                             DataWarehouse        * old_dw,
                             DataWarehouse        * new_dw);

    void testConservation(const ProcessorGroup  *,
                          const PatchSubset     * patches,
                          const MaterialSubset  *,
                          DataWarehouse         * old_dw,
                          DataWarehouse         * new_dw);

    void errorEstimate(const ProcessorGroup * pg,
                       const PatchSubset    * patches,
                       const MaterialSubset * matl,
                       DataWarehouse        * old_dw,
                       DataWarehouse        * new_dw,
                       bool initial);


    PassiveScalar(const PassiveScalar&);
    PassiveScalar& operator=(const PassiveScalar&);

    ProblemSpecP d_params;
    const Material* d_matl;
    MaterialSet*    d_matl_set;
    const MaterialSubset* d_matl_sub;

    //__________________________________
    //  Region used for initialization
    class Region {
    public:
      Region(GeometryPieceP piece, ProblemSpecP&);

      GeometryPieceP piece;
      double initialScalar;
      bool  sinusoidalInitialize;
      IntVector freq;
      bool  linearInitialize;
      Vector slope;
      bool cubicInitialize;
      Vector direction;
      bool quadraticInitialize;
      Vector coeff;
      bool exponentialInitialize_1D;
      bool exponentialInitialize_2D;
      bool triangularInitialize;

      bool  uniformInitialize;
    };

    //__________________________________
    //  For injecting a scalar inside the domain
    class interiorRegion {
    public:
      interiorRegion(GeometryPieceP piece, ProblemSpecP&);
      GeometryPieceP piece;
      double value;
      double clampValue;
    };

    //__________________________________
    //
    class Scalar {
    public:
      int index;
      std::string name;
      std::string fullName;

      // labels for this particular scalar
      VarLabel* Q_CCLabel;
      VarLabel* Q_src_CCLabel;
      VarLabel* mag_grad_Q_CCLabel;
      VarLabel* diffusionCoef_CCLabel;
      VarLabel* sum_Q_CCLabel;
      VarLabel* expDecayCoefLabel;

      std::vector<Region*> regions;
      std::vector<interiorRegion*> interiorRegions;

      double decayRate;               // constant decayRate
      double diff_coeff;
      double refineCriteria;

      // for exponential decay model
      double  c1 {-9};
      double  c2 {-9};
      double  c3 {-9};
      std::string c2_filename {"-9"};
    };

    Scalar* d_scalar;

    //__________________________________
    // global constants
    bool d_runConservationTask  {false};
    bool d_withExpDecayModel    {false};
    bool d_reinitializeDomain   {false};

    enum decayCoef{ constant, variable, none};
    decayCoef  d_decayCoef = none;

  };
}

#endif
