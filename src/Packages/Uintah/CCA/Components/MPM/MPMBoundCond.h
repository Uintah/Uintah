#ifndef UINTAH_MPMBOUNDCOND_H
#define UINTAH_MPMBOUNDCOND_H

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>


namespace Uintah {

using namespace SCIRun;

  class MPMBoundCond {

  public:
    
    MPMBoundCond();
    ~MPMBoundCond();

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<Vector>& variable,int n8or27=8);

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<double>& variable,int n8or27=8);

    void setBoundaryCondition(const Patch* patch,int dwi, const string& type,
			      NCVariable<double>& variable,
			      constNCVariable<double>& gvolume,int n8or27=8);

  private:

  };



}

#endif
