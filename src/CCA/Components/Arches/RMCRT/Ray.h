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
  class Ray{

    public: 

      Ray( const ArchesLabel* labels ); 
      ~Ray(); 

      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& inputdb ); 

      /** @brief Algorithm for tracing rays through a patch */ 
      void sched_rayTrace( const LevelP& level, SchedulerP& sched );
      void rayTrace( const ProcessorGroup* pc, 
          const PatchSubset* patches, 
          const MaterialSubset* matls, 
          DataWarehouse* old_dw, 
          DataWarehouse* new_dw ); 

      //  void VecUnitize( double direction_vector, double length);
      //  void VecLength(double length);

    private: 
      
      double _pi; 
      double _alpha;//absorptivity of the walls
      double d_Threshold;
      int d_NoOfRays;
      int _slice;
       // Arches labels
      const ArchesLabel* d_lab; 
      MTRand _mTwister; 
      int i,j,k;
      
      //double length;//this tells us the length of the direction vector.  It gets changed to unity during the VecUnitize function
      //double &length_ = length;//so we can pass length by reference
      //Vector &A_ = direction_vector; //so directionVector can be passed by reference to VecUnitize
      // Vector emiss_point;
      }; // class Ray
} // namespace Uintah

#endif
