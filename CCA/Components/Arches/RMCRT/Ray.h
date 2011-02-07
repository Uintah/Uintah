#ifndef Uintah_Component_Arches_Ray_h
#define Uintah_Component_Arches_Ray_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/RMCRT/MersenneTwister.h>
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

//==========================================================================

/**
 * @class Ray
 * @author Isaac Hunsaker
 * @date July 8, 2010 
 *
 * @brief This file traces N (usually 1000+) rays per cell until the intensity reaches a predetermined threshold
 *
 *
 */

namespace Uintah{

  class ArchesLabel; 
  class Ray  {

    public: 

      Ray( const ArchesLabel* labels ); 
      ~Ray(); 

      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& inputdb ); 

      /** @brief Algorithm for tracing rays through a patch */ 
      void sched_rayTrace( const LevelP& level, 
                           SchedulerP& sched );

      /** @brief Schedule compute of blackbody intensity */ 
      void sched_blackBodyIntensity( const LevelP& level, 
                                     SchedulerP& sched );

      /** @brief Initializes properties for the algorithm */ 
      void sched_initProperties( const LevelP&, SchedulerP& sched, int time_sub_step );

      /** @brief Give access to the flux divergence term */
      inline const VarLabel* getDivQLabel() { return divQ_label; };  

      //  void VecUnitize( double direction_vector, double length);
      //  void VecLength(double length);

    private: 
      
      double _pi;
      double _alpha;//absorptivity of the walls
      double _Threshold;
      double _sigma; 
      int    _NoOfRays;
      int    _slice;
      const double _sigma_over_pi; // Stefan Boltzmann divided by pi (W* m-2* K-4)

       // Arches labels
      const ArchesLabel* d_lab; 
      MTRand _mTwister; 
      int i,j,k;
      bool _benchmark_1; 

      const VarLabel* sigmaT4_label; 
      const VarLabel* divQ_label; 
      const VarLabel* d_blackBodyIntensityLabel; 

      //----------------------------------------
      void rayTrace( const ProcessorGroup* pc, 
          const PatchSubset* patches, 
          const MaterialSubset* matls, 
          DataWarehouse* old_dw, 
          DataWarehouse* new_dw ); 
      
      //----------------------------------------
      void initProperties( const ProcessorGroup* pc, 
          const PatchSubset* patches, 
          const MaterialSubset* matls, 
          DataWarehouse* old_dw, 
          DataWarehouse* new_dw,
          int time_sub_step ); 

      //----------------------------------------
      void blackBodyIntensity( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw );

      //double length;//this tells us the length of the direction vector.  It gets changed to unity during the VecUnitize function
      //double &length_ = length;//so we can pass length by reference
      //Vector &A_ = direction_vector; //so directionVector can be passed by reference to VecUnitize
      // Vector emiss_point;
      }; // class Ray
} // namespace Uintah

#endif
