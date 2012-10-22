/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef Packages_Uintah_CCA_Components_Examples_Interpolator_h
#define Packages_Uintah_CCA_Components_Examples_Interpolator_h

#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/NCVariable.h>



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
