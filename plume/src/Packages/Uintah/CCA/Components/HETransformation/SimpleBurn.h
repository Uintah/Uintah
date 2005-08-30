#ifndef __SIMPLE_BURN_H__
#define __SIMPLE_BURN_H__

#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>

namespace Uintah {
/**************************************

CLASS
   SimpleBurn
   
   Short description...

GENERAL INFORMATION

   SimpleBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Burn_Model_Simple

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class SimpleBurn : public Burn { 
    private:
      double thresholdTemp, thresholdPressure, Enthalpy, BurnCoeff;
      double refPressure;
      
      // Prevent copying of this class
      // copy constructor

      SimpleBurn(const SimpleBurn &burn);
      SimpleBurn & operator=(const SimpleBurn &burn);
      
    public:
      // Constructor
      SimpleBurn(ProblemSpecP& ps);
      
      // Destructor
      ~SimpleBurn();

      void computeBurn(double gasTemperature,
		       double gasPressure,
		       double materialMass,
		       double materialPressure,
		       double &burnedMass,
		       double &releasedHeat,
		       double &delT,
		       double &surfaceArea);

       double getThresholdTemperature();

    };
} // End namespace Uintah
    


#endif /* __SIMPLE_BURN_H__*/

