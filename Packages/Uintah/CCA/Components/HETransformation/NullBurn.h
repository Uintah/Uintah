#ifndef __NULL_BURN_H__
#define __NULL_BURN_H__

#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {
/**************************************

CLASS
   NullBurn
   
   Short description...

GENERAL INFORMATION

   NullBurn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Burn_Model_Null

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

    class NullBurn : public Burn { 
    private:
      
      // Prevent copying of this class
      // copy constructor
      NullBurn(const NullBurn &burn);
      NullBurn & operator=(const NullBurn &burn);
      
    public:
      // Constructor
      NullBurn(ProblemSpecP& ps);
      
      // Destructor
      ~NullBurn();

      void computeBurn(double gasTemperature,
		       double gasPressure,
		       double materialMass,
		       double materialTemperature,
		       double &burnedMass,
		       double &releasedHeat,
		       double &delT,
		       double &surfaceArea);

       double getThresholdTemperature();


    };
} // End namespace Uintah
    


#endif /* __NULL_BURN_H__*/
