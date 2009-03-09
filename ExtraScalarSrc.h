/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- ExtraScalarSrc.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ExtraScalarSrc_h
#define Uintah_Component_Arches_ExtraScalarSrc_h

#include <CCA/Components/Arches/Arches.h>

/**************************************
CLASS
   ExtraScalarSrc
   
   Class ExtraScalarSrc is an abstract base class

GENERAL INFORMATION
   ExtraScalarSrc.h - declaration of the class
   
   Author: Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   July 30th , 2007
   
   C-SAFE 
   
   Copyright U of U 2007

KEYWORDS


DESCRIPTION
   Class ExtraScalarSrc is an abstract type

WARNING
   none
****************************************/

namespace Uintah {
class TimeIntegratorLabel;
class ExtraScalarSrc {

public:

  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for ExtraScalarSrc.
  ExtraScalarSrc(const ArchesLabel* label, 
                 const MPMArchesLabel* MAlb,
                 const VarLabel* d_src_label);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for ExtraScalarSrc.
  virtual ~ExtraScalarSrc();


  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Interface for Set up of the problem specification database
  virtual void problemSetup(const ProblemSpecP& db) = 0;


  virtual void sched_addExtraScalarSrc(SchedulerP& sched, 
                                       const PatchSet* patches,
                                       const MaterialSet* matls,
                                       const TimeIntegratorLabel* timelabels) = 0;

  virtual const string getTableName() = 0;
  virtual void setTableIndex(int tableIndex) = 0;  
                                         
protected:
  const ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;
  const VarLabel* d_scalar_nonlin_src_label;

private:

}; // End class ExtraScalarSrc
} // End namespace Uintah

#endif


