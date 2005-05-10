//----- DORadiationModel.h --------------------------------------------------

#ifndef Uintah_Component_Models_DORadiationModel_h
#define Uintah_Component_Models_DORadiationModel_h

/***************************************************************************
CLASS
    DORadiationModel
       Sets up the DORadiationModel

       Inputs are temperature (gets modified), abskg (gets modified),
       sootFVIN, co2, and h2o.  Outputs are the fluxes and source term.
       
GENERAL INFORMATION
    DORadiationModel.h - Declaration of DORadiationModel class

    Original Authors: Gautham Krishnamoorthy (gautham@crsim.utah.edu) 
                      Rajesh Rawat (rawat@crsim.utah.edu)

    Modified for use in Models, April 2005 by Seshadri Kumar
    
    Creation Date : 06-18-2002

    C-SAFE
    
    Copyright U of U 2005

KEYWORDS

DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/
#include <Packages/Uintah/CCA/Components/Models/Radiation/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationModel.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/RadiationSolver.h>

namespace Uintah {

class DORadiationModel: public RadiationModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      DORadiationModel(const ProcessorGroup* myworld);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for radiation model
      //
      ~DORadiationModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      void problemSetup(const ProblemSpecP& params);
    
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////

      void computeRadiationProps(const ProcessorGroup* pc,
					 const Patch* patch,
					 CellInformation* cellinfo,
					 RadiationVariables* vars,
					 RadiationConstVariables* constvars);
      //
      /////////////////////////////////////////////////////////////////////////
      

      void boundaryCondition(const ProcessorGroup* pc,
			     const Patch* patch,
			     RadiationVariables* vars);

      ////////////////////////////////////////////////////////////////////////

      void intensitysolve(const ProcessorGroup* pc,
				  const Patch* patch,
				  CellInformation* cellinfo, 
				  RadiationVariables* vars,
				  RadiationConstVariables* constvars);
      ////////////////////////////////////////////////////////////////////////

protected: 

private:

      double d_xumax;
      
      int d_sn, d_totalOrds; // totalOrdinates = sn*(sn+2)

      void computeOrdinatesOPL();

      int MAXITR;
      double QACCU, d_opl, af, qerr, totsrc;
      int iflag, iriter;
      int lambda;
      double wavemin, wavemax, dom, omega, srcsum;
      int ffield;
      int wall;
      int symtry;
      int pfield;
      int sfield;
      int pbcfld;
      int outletfield;
      bool d_SHRadiationCalc, lprobone, lprobtwo, lprobthree, lradcal, lwsgg, lplanckmean, lpatchmean;

      OffsetArray1<double> fraction;
      OffsetArray1<double> fractiontwo;

      //      OffsetArray1<double> ord;
      OffsetArray1<double> oxi;
      OffsetArray1<double> omu;
      OffsetArray1<double> oeta;
      OffsetArray1<double> wt;
      OffsetArray1<double> arean;
      OffsetArray1<double> areatb;

      OffsetArray1<double> rgamma;
      OffsetArray1<double> sd15;
      OffsetArray1<double> sd;
      OffsetArray1<double> sd7;
      OffsetArray1<double> sd3;

      OffsetArray1<double> srcbm;
      OffsetArray1<double> srcpone;
      OffsetArray1<double> qfluxbbm;
      const ProcessorGroup* d_myworld;


}; // end class RadiationModel

} // end namespace Uintah

#endif





