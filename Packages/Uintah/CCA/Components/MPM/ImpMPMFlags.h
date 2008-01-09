#ifndef __IMP_MPM_FLAGS_H__
#define __IMP_MPM_FLAGS_H__

#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ImpMPMFlags
    \brief A structure that store the flags used for a MPM simulation
    \author John Schmidt \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2007 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////


  class UINTAHSHARE ImpMPMFlags : public MPMFlags {

  public:

    ImpMPMFlags();

    virtual ~ImpMPMFlags();

    virtual void readMPMFlags(ProblemSpecP& ps);
    virtual void outputProblemSpec(ProblemSpecP& ps);


    bool d_projectHeatSource;
    bool d_doMechanics;
    double  d_conv_crit_disp;
    double  d_conv_crit_energy;
    bool d_dynamic;
    int d_max_num_iterations;
    int d_num_iters_to_decrease_delT;
    int d_num_iters_to_increase_delT;
    double d_delT_decrease_factor;
    double d_delT_increase_factor;
    string d_solver_type;
    bool d_temp_solve;
    bool d_interpolateParticleTempToGridEveryStep;

  private:

    ImpMPMFlags(const ImpMPMFlags& state);
    ImpMPMFlags& operator=(const ImpMPMFlags& state);
    
  };

} // End namespace Uintah

#endif  // __IMP_MPM_FLAGS_H__ 
