#ifndef __TST_H__
#define __TST_H__

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include "EquationOfState.h"

namespace Uintah {
/**************************************

CLASS
   EquationOfState
   
   A version of the Twu-Sim-Tassone equation of state, as described below:

        RT         a
   P = ---- - ------------
       v-b    (v+ub)(v+wb)

   P = pressure 
   R = gas constant
   T = temperature
   v = specific volume
   a, b, u, w are constants

GENERAL INFORMATION

   TST.h

   Changwei Xiong
   Department of Chemistry
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Equation_of_State TST

DESCRIPTION
   Long description...
  
WARNING
****************************************/

      class TST : public EquationOfState {
      public:

        TST(ProblemSpecP& ps);
        virtual ~TST();

        virtual void outputProblemSpec(ProblemSpecP& ps);
        
        virtual double computeRhoMicro(double press,double gamma,
                                      double cv, double Temp, double rho_guess);
         
        virtual void computePressEOS(double rhoM, double gamma,
                                     double cv, double Temp,
                                     double& press, double& dp_drho,
                                     double& dp_de);

        virtual void computeTempCC(const Patch* patch,
                                   const string& comp_domain,
                                   const CCVariable<double>& press, 
                                   const CCVariable<double>&,
                                   const CCVariable<double>& cv,
                                   const CCVariable<double>& rhoM, 
                                   CCVariable<double>& Temp,
                                   Patch::FaceType face);
       

        virtual double getAlpha(double Temp,double sp_vol, double P, double cv);
         
        virtual void hydrostaticTempAdjustment(Patch::FaceType, 
                                               const Patch*,
                                               const vector<IntVector>&,
                                               Vector&,
                                               const CCVariable<double>&,
                                               const CCVariable<double>&,
                                               const Vector&,
                                               CCVariable<double>&);

      private:
        double   a;
        double   b;
        double   u;
        double   w;
	double   Gamma;
	
	/* The following used only in TST::computeRhoMicro */
	double   Pressure;
	double   Temperature;
	double   SpecificHeat;
	double   IL, IR;

	double func(double rhoM);
	double deri(double rhoM);
	void   setInterval(double f, double rhoM);

      };
} // End namespace Uintah
      
#endif  // __TST_H__


