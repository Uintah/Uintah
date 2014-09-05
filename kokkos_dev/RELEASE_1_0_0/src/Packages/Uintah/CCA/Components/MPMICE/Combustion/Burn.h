#ifndef __BURN_H__
#define __BURN_H__




namespace Uintah {

   class MPMICELabel;

/**************************************

CLASS
   Burn 
   
   Short description...

GENERAL INFORMATION

   Burn.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Burn_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

      class Burn {
      public:
         // Constructor
	 Burn();
	 virtual ~Burn();

	 // Basic burn methods

	  bool isBurnable();

	  virtual void computeBurn(double gasTemperature,
				   double gasPressure,
				   double materialMass,
				   double materialTemp,
				   double &burnedMass,
				   double &releasedHeat) = 0;

       protected:
	  bool d_burnable;
	  MPMICELabel* lb;
    
      };
      
} // End namespace Uintah

#endif // __Burn_H__



