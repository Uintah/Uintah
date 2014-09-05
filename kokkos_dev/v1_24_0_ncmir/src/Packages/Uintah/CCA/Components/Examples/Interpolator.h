#ifndef Packages_Uintah_CCA_Components_Examples_Interpolator_h
#define Packages_Uintah_CCA_Components_Examples_Interpolator_h

#include <Packages/Uintah/Core/Parallel/UintahParallelComponent.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>



/**************************************

CLASS
   Interpolator
   
   Interpolator simulation

GENERAL INFORMATION

   Interpolator.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Interpolator

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/


namespace Uintah {


class Interpolator
{
private:
    IntVector refine_support_;
    IntVector coarsen_support_;
    int max_refine_support_;
    int max_coarsen_support_;
    int factor_;

public:
    enum PointType 
    {
        Inner,
	East,
	West,
	North,
	South,
	Up,
	Down
    };

    Interpolator(int factor);
    IntVector getSupportRef() { return refine_support_; }
    int getXSupportRefine() { return refine_support_[0]; }
    int getYSupportRefine() { return refine_support_[1]; }
    int getZSupportRefine() { return refine_support_[2]; }
    int getMaxSupportRefine() { return max_refine_support_; }
    double refine(constNCVariable<double>& variable, IntVector index, Interpolator::PointType type);
    double refine(constNCVariable<double>& variable1, double weight1,
		  constNCVariable<double>& variable2, double weight2,
		  IntVector index, Interpolator::PointType type);

    IntVector getSupportCoarsen() { return coarsen_support_; }
    int getXSupportCoarsen() { return coarsen_support_[0]; }
    int getYSupportCoarsen() { return coarsen_support_[1]; }
    int getZSupportCoarsen() { return coarsen_support_[2]; }
    int getMaxSupportCoarsen() { return max_coarsen_support_; }
    double coarsen(const NCVariable<double>& variable, IntVector index, Interpolator::PointType type);

    IntVector fineToCoarseIndex(const IntVector& index) { return index / IntVector(factor_,factor_,factor_); }
    IntVector coarseToFineIndex(const IntVector& index) { return index * IntVector(factor_,factor_,factor_); }
};

}

#endif
