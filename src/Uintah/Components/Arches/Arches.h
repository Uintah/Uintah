
#ifndef Uintah_Component_Arches_Arches_h
#define Uintah_Component_Arches_Arches_h

/*
 * Placeholder - nothing here yet
 */

#include <Uintah/Parallel/UintahParallelComponent.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Parallel/ProcessorContext.h>

namespace Uintah {
namespace Components {
  using Uintah::Interface::CFDInterface;
  using Uintah::Parallel::UintahParallelComponent;
  using Uintah::Interface::ProblemSpecP;
  using Uintah::Grid::GridP;
  using Uintah::Grid::LevelP;
  using Uintah::Grid::Region;
  using Uintah::Interface::DataWarehouseP;
  using Uintah::Interface::SchedulerP;
  using Uintah::Parallel::ProcessorContext;

class PhysicalConstants;
class NonlinearSolver;
class Properties;
class TurbulenceModel;
class BoundaryCondition;

class Arches : public UintahParallelComponent, public CFDInterface {
public:
    static const int NDIM = 3;

    Arches();
    virtual ~Arches();

    virtual void problemSetup(const ProblemSpecP& params, GridP& grid,
			      DataWarehouseP&);

    virtual void problemInit(const LevelP& level,
			     SchedulerP& sched, DataWarehouseP& dw,
			     bool restrt);
    virtual void computeStableTimestep(const LevelP& level,
				       SchedulerP&, DataWarehouseP&);
    virtual void timeStep(double t, double dt,
			  const LevelP& level, SchedulerP&,
			  const DataWarehouseP&, DataWarehouseP&);
    void sched_paramInit(const LevelP& level,
			 SchedulerP& sched, DataWarehouseP& dw);

private:
    Arches(const Arches&);
    Arches& operator=(const Arches&);
    void paramInit(const ProcessorContext*,
		   const Region* region,
		   const DataWarehouseP& old_dw);
    double d_deltaT;
    PhysicalConstants* d_physicalConsts;
    NonlinearSolver* d_nlSolver;
  // properties...solves density, temperature and species concentrations
    Properties* d_props;
  
  // Turbulence Model
    TurbulenceModel* d_turbModel;
  // Boundary conditions
    BoundaryCondition* d_boundaryCondition;
};

} // end namespace Components
} // end namespace Uintah

//
// $Log$
// Revision 1.11  2000/04/11 19:55:52  rawat
// modified nonlinear solver for initialization
//
// Revision 1.10  2000/04/07 23:02:16  sparker
// Fixed arches compile
//
// Revision 1.9  2000/04/07 18:30:12  rawat
// Added problem initialization function in Arches.cc
//
// Revision 1.8  2000/03/23 20:05:13  jas
// Changed the location of ProblemSpec from Grid to Interface in the include
// file path.
//
// Revision 1.7  2000/03/21 21:27:03  dav
// namespace fixs
//
//

#endif

