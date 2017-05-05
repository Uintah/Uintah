/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//------------------------ CQMOM.h -----------------------------------


#ifndef Uintah_Components_Arches_CQMOM_h
#define Uintah_Components_Arches_CQMOM_h


//NOTE: I just listed all includes that DQMOM uses for now. As this is finalized some can likely be removed.
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

#include <map>

#include <sci_defs/cuda_defs.h>

namespace Uintah {
  
  //-------------------------------------------------------
  
  /**
   * @class    CQMOM
   * @author   Alex Abboud
   * @date     May 2014 first iteration, adapted from DQMOM class
   *
   * @brief    This class is a wrapper for the CQMOM inversion problem, it will take in the moments of the distribution
   *           and output the weights and abscissas to be used in transport and source terms
   *
   * @details  This class does not actually calculate the weights and abscissas. It will take care of the task scheduling and
   *           requirements that need to be met in order to take care of the lists of moments, weights and absicissas requried.
   *           The CQMOMInversion function is where the actual calculation is done.
   *
   */
  
  //-------------------------------------------------------
  
  class ArchesLabel;
  class CQMOMEqn;
  class ModelBase;
  
  class CQMOM {
    
  public:
    
    CQMOM( ArchesLabel* fieldLabels, bool usePartVel );
    
    ~CQMOM();
    
    typedef std::vector<int> MomentVector;
    
    /** @brief Obtain parameters from input file and process */
    void problemSetup( const ProblemSpecP& params );
    
    //NOTE: here are the new methods for CQMOM, these should mirror some of dqmom methods
    /** @brief  Schedule the CQMOM inversion problem
     */
    void sched_solveCQMOMInversion( const LevelP & level,
                                    SchedulerP   & sched,
                                    int          timesubstep);
    
    /** @brief actually solve the inversion and put weights and abscissas in datawarehouse
     */
    void solveCQMOMInversion( const ProcessorGroup *,
                              const PatchSubset    * patches,
                              const MaterialSubset *,
                              DataWarehouse        * old_dw,
                              DataWarehouse        * new_dw);
    
    //Other permutations of the problem, possibly find a way to condense all of these
    /** @brief  Schedule the CQMOM inversion problem with conditioning 3|2|1
     */
    void sched_solveCQMOMInversion321( const LevelP & level,
                                       SchedulerP   & sched,
                                       int          timesubstep);
    
    /** @brief actually solve the inversion and put weights and abscissas in datawarehouse
     */
    void solveCQMOMInversion321( const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset *,
                                 DataWarehouse        * old_dw,
                                 DataWarehouse        * new_dw);
    
    /** @brief  Schedule the CQMOM inversion problem with conditioning 3|1|2
     */
    void sched_solveCQMOMInversion312( const LevelP & level,
                                       SchedulerP   & sched,
                                       int          timesubstep);
    
    /** @brief actually solve the inversion and put weights and abscissas in datawarehouse
     */
    void solveCQMOMInversion312( const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset *,
                                 DataWarehouse        * old_dw,
                                 DataWarehouse        * new_dw);
    
    /** @brief  Schedule the CQMOM inversion problem with conditioning 2|1|3
     */
    void sched_solveCQMOMInversion213( const LevelP & level,
                                       SchedulerP   & sched,
                                       int          timesubstep);
    
    /** @brief actually solve the inversion and put weights and abscissas in datawarehouse
     */
    void solveCQMOMInversion213( const ProcessorGroup *,
                                 const PatchSubset    * patches,
                                 const MaterialSubset *,
                                 DataWarehouse        * old_dw,
                                 DataWarehouse        * new_dw);
    
    
    /** @brief Schedule re-calculation of moments */
    void sched_momentCorrection( const LevelP & level,
                                 SchedulerP   & sched,
                                 int          timeSubStep );
    
    /** @brief re-calculate moments as needed, if unrealizable abscissas occur */
    void momentCorrection( const ProcessorGroup *,
                           const PatchSubset    * patches,
                           const MaterialSubset *,
                           DataWarehouse        * old_dw,
                           DataWarehouse        * new_dw );
    
    inline bool getOperatorSplitting() {return d_doOperatorSplitting; };
    
    inline bool getPartVel() {return d_usePartVel; };
    
    inline int getUVelIndex() {return uVelIndex; };
    
    inline int getVVelIndex() {return vVelIndex; };
    
    inline int getWVelIndex() {return wVelIndex; };

//____________________________
    
  private:
    
    //NOTE: should the CQMOMInversion be a private function that exists here, instead of own file/class?
    //if so make private fucntions for wheeler & vandermonde algorithms exist here as well
    // ie void vandermonde(), void wheeler(), void inversion()
    
    std::vector<MomentVector> momentIndexes;     ///< Vector containing all moment indices
    
    std::vector<CQMOMEqn* > momentEqns;          ///< moment equation labels, (same order as input?)

    ArchesLabel* d_fieldLabels;
    
    int M;                                      // Number of internal coordiantes
    std::vector<int> N_i;                       //Number of quadrature nodes per internal coordinate
    std::vector<int> maxInd;                    //maximum moment order for each internal coordinate
    int nNodes;                                 //total number of quad nodes
    int momentSize;                             //how large to allocate flat array of moments
    int nMoments;                               //numebr of transported moments
    double d_small;                             //below this value of 0th moment assume all weights/abscissa = 0
    
    int d_timeSubStep;
    bool d_adaptive;                            //boolean to use adaptive number of nodes
    bool d_useLapack;                           //boolean to use lapack or vandermonde solver
    bool d_doOperatorSplitting;                 //use operator splitting to calculate nodes for multiple permutations
    bool d_usePartVel;                          //use particle velocity as an IC
    
    double weightRatio;                         //adaptive double for minimum allowed weigth ratio
    double abscissaRatio;                       //adaptive double for minimum allowed abscissa ratio
    
    std::vector<std::string> coordinateNames;   //list of internal coordiante names
    std::vector<std::string> weightNames;       //list of wieght names - to be used for dw->get
    std::vector<std::string> abscissaNames;     //list of absicassa names - to be used for dw->get
    
    std::vector<std::string> varTypes;

    int uVelIndex;
    int vVelIndex;
    int wVelIndex;
    
    // Clipping:
    struct ClipInfo{

      bool activated;                 ///< Clipping on/off for this internal coordinate
      bool do_low;                    ///< Do clipping on a min
      bool do_high;                   ///< Do clipping on a max
      bool clip_to_zero;              //clip to 0 instead of teh actual high/low
      double weight_clip;             //limit on the weight to apply zero-clipping
      double low;                     ///< Low clipping value
      double high;                    ///< High clipping value
      double tol;                     ///< Tolerance value for the min and max
      
    };
    
    std::vector<ClipInfo> clipNodes;
    
  }; // end class CQMOM
  
} // end namespace Uintah

#endif
