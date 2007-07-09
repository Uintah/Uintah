#ifndef UINTAH_MPMBOUNDCOND_H
#define UINTAH_MPMBOUNDCOND_H

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>

#include <Packages/Uintah/CCA/Components/MPM/share.h>

namespace Uintah {

using namespace SCIRun;

  class SCISHARE MPMBoundCond {

  public:
    
    MPMBoundCond();
    ~MPMBoundCond();

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<Vector>& variable,
                              string interp_type="linear");

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<double>& variable,
                              string interp_type="linear");

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<double>& variable,
			      constNCVariable<double>& gvolume,
                              string interp_type="linear");

  private:

  };



}

#endif
