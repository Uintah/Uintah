#ifndef Uintah_Component_Arches_PropertyModel_h
#define Uintah_Component_Arches_PropertyModel_h

/**
* @class PropertyModel
* @author Xiaojing Sun
* @date Dec 11, 2008
*
* @brief Radiative Property Models
*
*
*/


#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationSolver.h>

namespace Uintah {

class BoundaryCondition;

class PropertyModel: public RadiationModel {

public:

  PropertyModel();

  virtual ~PropertyModel();
  
  /** @brief Compute Radiation Participating media properties */
  virtual void computeRadiationProps(const ProcessorGroup* pc,
				     const Patch* patch,
				     CellInformation* cellinfo,
				     ArchesVariables* vars,
				     ArchesConstVariables* constvars);

}

}

