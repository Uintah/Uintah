/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_HOMEBREW_VarTypes_H
#define UINTAH_HOMEBREW_VarTypes_H


#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Disclosure/TypeUtils.h>

namespace Uintah {
   /**************************************
     
     CLASS
       VarTypes
      
       Short Description...
      
     GENERAL INFORMATION
      
       VarTypes.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
     KEYWORDS
       VarTypes
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/

   // System vars related to the application
   const std::string timeStep_name( "timeStep" );
   const std::string simTime_name( "simulationTime" );
   const std::string delT_name( "delT" );

   const std::string outputInterval_name( "outputInterval" );
   const std::string checkpointInterval_name( "checkpointInterval" );
   const std::string outputTimeStep_name( "outputTimeStep" );
   const std::string outputPreviousTimeStep_name( "outputPreviousTimeStep" );
   const std::string checkpointTimeStep_name( "checkpointTimeStep" );
   const std::string checkpointPreviousTimeStep_name( "checkpointPreviousTimeStep" );
   const std::string recomputeTimeStep_name( "recomputeTimeStep" );
   const std::string abortTimeStep_name( "abortTimeStep" );
   const std::string endSimulation_name( "endSimulation" );
  
   typedef SoleVariable<unsigned int > timeStep_vartype;
   typedef SoleVariable<double       >  simTime_vartype;
  
   typedef ReductionVariable<double, Reductions::Min<double> > delt_vartype;

   typedef ReductionVariable<double, Reductions::Max<double> > max_vartype;
   
   typedef ReductionVariable<double, Reductions::Min<double> > min_vartype;

   typedef ReductionVariable<double, Reductions::Sum<double> > sum_vartype;

   typedef ReductionVariable<bool,   Reductions::And<bool> >   bool_and_vartype;

   typedef ReductionVariable<bool,   Reductions::Or<bool> >    bool_or_vartype;
    
   typedef ReductionVariable<Vector, Reductions::Min<Vector> > minvec_vartype;
   
   typedef ReductionVariable<Vector, Reductions::Max<Vector> > maxvec_vartype;
   
   typedef ReductionVariable<Vector, Reductions::Sum<Vector> > sumvec_vartype;

   typedef ReductionVariable<long64, Reductions::Sum<long64> > sumlong_vartype;

   typedef ReductionVariable<long64, Reductions::Sum<long long> > sumlonglong_vartype;

} // End namespace Uintah

#endif
