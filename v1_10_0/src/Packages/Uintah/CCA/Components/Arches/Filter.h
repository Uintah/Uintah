
#ifndef Uintah_Components_Arches_Filter_h
#define Uintah_Components_Arches_Filter_h

#include <sci_defs.h>

#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/CCA/Ports/SchedulerP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Core/Geometry/IntVector.h>

#include <Core/Containers/Array1.h>

#ifdef HAVE_PETSC
extern "C" {
#include "petscmat.h"
}
#endif

namespace Uintah {

class ProcessorGroup;
 class ArchesLabel;

using namespace SCIRun;
class BoundaryCondition;


/**************************************
CLASS
   Filter
   
   Class Filter uses petsc's matmult operation
   solver

GENERAL INFORMATION
   Filter.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   
   Creation Date:   July 10, 2002
   
   C-SAFE 
   
   Copyright U of U 2002

KEYWORDS


DESCRIPTION
   Class Filter uses petsc matmult operation for applying box filter

WARNING
   none

****************************************/
class Filter {

public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Construct an instance of a Filter.
      Filter(const ArchesLabel* label,
	     BoundaryCondition* bndryCondition,
	     const ProcessorGroup* myworld);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Destructor
      ~Filter();

      // GROUP: Problem Setup:
      ////////////////////////////////////////////////////////////////////////
      // Problem setup
      void problemSetup(const ProblemSpecP& params);

      bool isInitialized() {
	return d_matrixInitialize;
      }
      void sched_buildFilterMatrix(const LevelP& level,
				   SchedulerP& sched);

      void buildFilterMatrix(const ProcessorGroup* pg,
			     const PatchSubset* patches,
			     const MaterialSubset*,
			     DataWarehouse*,
			     DataWarehouse* new_dw);

      // constructs filter matrix will be different for different types of filters
     
      void matrixCreate(const PatchSet* allpatches,
			const PatchSubset* mypatches);
      void setFilterMatrix(const ProcessorGroup* pc, const Patch* patch,
			   CellInformation* cellinfo, constCCVariable<int>& cellType);
      bool applyFilter(const ProcessorGroup* pc, const Patch* patch,
		       Array3<double>& var, Array3<double>& filterVar);
      bool applyFilter(const ProcessorGroup* pc, const Patch* patch,
		       constCCVariable<double>& var, Array3<double>& filterVar);
      void destroyMatrix();
protected:

private:
   const ProcessorGroup* d_myworld;
   const PatchSet* d_perproc_patches;
   const ArchesLabel* d_lab;
   BoundaryCondition* d_boundaryCondition;

   bool d_matrixInitialize;
#ifdef HAVE_PETSC
   map<const Patch*, int> d_petscGlobalStart;
   map<const Patch*, Array3<int> > d_petscLocalToGlobal;
   Mat A;
   Vec d_x, d_b;
   int d_nz, o_nz; // number of non zero values in a row
#endif
}; // End class Filter.h
} // End namespace Uintah

#endif  
  





