#ifndef UINTAH_MPMBOUNDCOND_H
#define UINTAH_MPMBOUNDCOND_H

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>


namespace Uintah {

using namespace SCIRun;

  class MPMBoundCond {

  public:
    
    MPMBoundCond();
    ~MPMBoundCond();

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<Vector>& variable);

    void setBoundaryConditionJohn(const Patch* patch,int dwi, 
				  const string& type,
				  NCVariable<Vector>& variable);

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<double>& variable);

    void setBoundaryConditionJohn(const Patch* patch,int dwi, 
				  const string& type,
				  NCVariable<double>& variable);


    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<double>& variable,
			      constNCVariable<double>& gvolume);

    void setBoundaryConditionJohn(const Patch* patch,int dwi, 
				  const string& type,
				  NCVariable<double>& variable,
				  constNCVariable<double>& gvolume);
    
  private:

  };



}

#endif
