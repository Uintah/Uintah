#ifndef ExplicitTimeInt_h
#define ExplicitTimeInt_h
#include <Core/Grid/LevelP.h>
#include <CCA/Ports/SimulationInterface.h>
#include <Core/Grid/SimulationStateP.h>

//===========================================================================

namespace Uintah {

using namespace SCIRun; 
class Fields;   
class ExplicitTimeInt {
    
public:
    
    ExplicitTimeInt(const Fields* fieldLabels);

    ~ExplicitTimeInt(); 
    /** @brief Input file interface and constant intialization */ 
    void problemSetup(const ProblemSpecP& params);
   /** @brief A template forward Euler update for a single 
               variable for a single patch */ 
    template <class phiT, class constphiT>
    void singlePatchFEUpdate( const Patch* patch, 
                              phiT& phi, 
                              constphiT& RHS, 
                              double dt );
    /** @brief A template for time averaging using a Runge-kutta form */  
    template <class phiT, class constphiT>
    void timeAvePhi( const Patch* patch, 
                     phiT& phi, 
                     constphiT& old_phi, 
                     int step, Vector alpha, Vector beta ); 


    Vector d_beta, d_alpha; 

private:
    const Fields* d_fieldLabels;
    int d_step;

  }; //end Class ExplicitTimeInt
  

  template <class phiT, class constphiT>
  void ExplicitTimeInt::singlePatchFEUpdate( const Patch* patch, 
                                             phiT& phi, 
                                             constphiT& RHS, 
                                             double dt )
  {

    Vector dx = patch->dCell();

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){

      double vol = dx.x()*dx.y()*dx.z();

      phi[*iter] += dt/vol*(RHS[*iter]);

    } 
  }

  template <class phiT, class constphiT>
  void ExplicitTimeInt::timeAvePhi( const Patch* patch, 
                                    phiT& phi, 
                                    constphiT& old_phi, 
                                    int step, Vector alpha, Vector beta )
  {
  
		for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
  
			phi[*iter] = d_alpha[step]*old_phi[*iter] + d_beta[step]*phi[*iter];	

		}
  }
} //end namespace Uintah
    
#endif

