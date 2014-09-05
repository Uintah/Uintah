#ifndef ExplicitTimeInt_h
#define ExplicitTimeInt_h
#include <Core/Grid/LevelP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>

//===========================================================================

namespace Uintah {

/** @class    ExplicitTimeInt
  * @author   Jeremy Thornock
  *
  */

using namespace SCIRun; 
class ArchesLabel;   
class ExplicitTimeInt {
    
public:
    
    ExplicitTimeInt(const ArchesLabel* fieldLabels);

    ~ExplicitTimeInt(); 
    /** @brief Input file interface and constant intialization */ 
    void problemSetup(const ProblemSpecP& params);
   /** @brief A template forward Euler update for a single 
               variable for a single patch */ 
    template <class phiT, class constphiT>
    void singlePatchFEUpdate( const Patch* patch, 
                              phiT& phi, 
                              constphiT& RHS, 
                              double dt, double time, 
                              const string eqnName );
   /** @brief A template forward Euler update for a single 
               variable for a single patch */ 
    template <class phiT, class constphiT>
    void singlePatchFEUpdate( const Patch* patch, 
                              phiT& phi, constCCVariable<double>& old_den, 
                              constCCVariable<double>& new_den, 
                              constphiT& RHS, 
                              double dt, double time,
                              const string eqnName );
    /** @brief A template for time averaging using a Runge-kutta form without density*/  
    template <class phiT, class constphiT>
    void timeAvePhi( const Patch* patch, 
                     phiT& phi, 
                     constphiT& old_phi, 
                     int step, double time );

    /** @brief A template for time averaging using a Runge-kutta form with density */ 
    template <class phiT, class constphiT>
    void timeAvePhi( const Patch* patch, 
                     phiT& phi, 
                     constphiT& old_phi, 
                     constphiT& old_den, 
                     constphiT& new_den, 
                     int step, double time ); 

    Vector ssp_beta, ssp_alpha; 
    Vector time_factor; 

    double d_LinfError; 
    double d_LinfSol; 

    std::string d_time_order; 

private:
    const ArchesLabel* d_fieldLabels;
    int d_step;

  }; //end Class ExplicitTimeInt
  
  // no density
  template <class phiT, class constphiT>
  void ExplicitTimeInt::singlePatchFEUpdate( const Patch* patch, 
                                             phiT& phi, 
                                             constphiT& RHS, 
                                             double dt, double time, 
                                             const string eqnName )
  {

#ifdef VERIFY_TIMEINT
    cout << "**********************************************************************" << endl;
    cout << endl;
    cout << "NOTICE! Using time integrator verification procedure." << endl;
    cout << "current equation: " << eqnName << endl; 
    cout << endl;

    // This is a verfication test on the time integrator only (no spatial error)
    double pi = acos(-1.0); 
    double RHS_test = cos(2.0*pi*time); 
#endif 


    Vector dx = patch->dCell();
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      double vol = dx.x()*dx.y()*dx.z();

#ifdef VERIFY_TIMEINT
      phi[c] += dt/vol*(RHS_test);
#else
      phi[c] += dt/vol*(RHS[c]);
#endif
    } 
  }

  // with density
  template <class phiT, class constphiT>
  void ExplicitTimeInt::singlePatchFEUpdate( const Patch* patch, 
                                             phiT& phi, constCCVariable<double>& old_den, 
                                             constCCVariable<double>& new_den, 
                                             constphiT& RHS, 
                                             double dt, double time, 
                                             const string eqnName )
 
  {


#ifdef VERIFY_TIMEINT
    cout << "**********************************************************************" << endl;
    cout << endl;
    cout << "NOTICE! Using time integrator verification procedure." << endl;
    cout << "current equation: " << eqnName << endl; 
    cout << "current time: " << time << endl;
    cout << endl;

    // This is a verfication test on the time integrator only (no spatial error)
    double pi = acos(-1.0); 
    double RHS_test = cos(2.0*pi*time); 
#endif 


    Vector dx = patch->dCell();
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      double vol = dx.x()*dx.y()*dx.z();

      // (rho*phi)^{t+\Delta t} = (rho*phi)^{t} + RHS
#ifdef VERIFY_TIMEINT
      phi[c] = old_den[c]*phi[c] + dt/vol*(RHS_test);
#else
      phi[c] = old_den[c]*phi[c] + dt/vol*(RHS[c]); 
#endif

      // phi^{t+\Delta t} = ((rho*phi)^{t} + RHS) / rho^{t + \Delta t} 
      phi[c] = phi[c] / new_den[c]; 

      double small = 1e-16; 
      if (new_den[c] < small) 
        phi[c] = 0.0; 

    } 
  }

//---------------------------------------------------------------------------
// Time averaging W/O density
//---------------------------------------------------------------------------
// ----RK AVERAGING
//     to get the time averaged phi^{time averaged}
//     See: Gettlieb et al., SIAM Review, vol 43, No 1, pp 89-112
//          Strong Stability-Preserving High-Order Time Discretization Methods
  template <class phiT, class constphiT>
  void ExplicitTimeInt::timeAvePhi( const Patch* patch, 
                                    phiT& phi, 
                                    constphiT& old_phi, 
                                    int step, double time )
  {

		for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
			phi[*iter] = ssp_alpha[step]*old_phi[*iter] + ssp_beta[step]*phi[*iter];	
    }


#ifdef VERIFY_TIMEINT
    // This computes the L_inf norm of the error for the integrator test. 
    if (step == 0 && d_time_order == "first" || 
        step == 1 && d_time_order == "second" || 
        step == 2 && d_time_order == "third") {
      d_LinfError = 0.0;
      d_LinfSol   = 0.0; 
    }  

    double error = 0.0;
    double pi = acos(-1.0); 
    Vector dx = patch->dCell();
    double exact = 1./(2.*pi)*sin(2.*pi*time);
    d_LinfSol = max(d_LinfSol, exact); 
    exact *= dx.x()*dx.y()*dx.z(); 
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      d_LinfError = max(d_LinfError, abs(phi[c] - exact));
    }
    error = d_LinfError / d_LinfSol; //normalize

    cout << "Error from time integration = " << error << endl;
    cout << endl;
    cout << "**********************************************************************" << endl;
#endif  


  }

//---------------------------------------------------------------------------
// Time averaging With Density
//---------------------------------------------------------------------------
// ----RK AVERAGING
//     to get the time averaged phi^{time averaged}
//     See: Gettlieb et al., SIAM Review, vol 43, No 1, pp 89-112
//          Strong Stability-Preserving High-Order Time Discretization Methods
  template <class phiT, class constphiT>
  void ExplicitTimeInt::timeAvePhi( const Patch* patch, 
                                    phiT& phi, 
                                    constphiT& old_phi, 
                                    constphiT& new_den, 
                                    constphiT& old_den, 
                                    int step, double time )
  {

		for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      double pred_density = ssp_alpha[step]*old_den[c] + ssp_beta[step]*new_den[c]; 

      if ( pred_density > 0 ) { 
			  phi[*iter] = ( ssp_alpha[step] * (old_den[c] * old_phi[c])
                   +   ssp_beta[step]  * (new_den[c] * phi[c]) ) / pred_density;	
      }

    }
  }

} //end namespace Uintah
    
#endif

