#ifndef UINTAH_MPMBOUNDCOND_H
#define UINTAH_MPMBOUNDCOND_H

#include <SCIRun/Core/Geometry/Vector.h>
#include <Core/Grid/Variables/NCVariable.h>

#include <CCA/Components/MPM/uintahshare.h>

namespace Uintah {

using namespace SCIRun;

  class UINTAHSHARE MPMBoundCond {

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
