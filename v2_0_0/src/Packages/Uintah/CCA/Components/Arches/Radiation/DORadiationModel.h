//----- DORadiationModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_DORadiationModel_h
#define Uintah_Component_Arches_DORadiationModel_h

/***************************************************************************
CLASS
    DORadiationModel
       Sets up the DORadiationModel
       
GENERAL INFORMATION
    DORadiationModel.h - Declaration of DORadiationModel class

    Author:Gautham Krishnamoorthy (gautham@crsim.utah.edu) 
           Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 06-18-2002

    C-SAFE
    
    Copyright U of U 2002

KEYWORDS

DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationModel.h>

namespace Uintah {

class BoundaryCondition;

class DORadiationModel: public RadiationModel {

public:

      // GROUP: Constructors:
      ///////////////////////////////////////////////////////////////////////
      //
      // Constructor taking
      //   [in] 
      //
      DORadiationModel(BoundaryCondition* bndry_cond, const ProcessorGroup* myworld);

      // GROUP: Destructors :
      ///////////////////////////////////////////////////////////////////////
      //
      // Virtual destructor for radiation model
      //
      virtual ~DORadiationModel();

      // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      //
      // Set up the problem specification database
      //
      virtual void problemSetup(const ProblemSpecP& params);
    
      // GROUP: Actual Action Methods :
      ///////////////////////////////////////////////////////////////////////
      /*
      virtual void computeRadiationProps(const ProcessorGroup* pc,
      				 const Patch* patch,
      				 CellInformation* cellinfo,
      				 ArchesVariables* vars);
      */
      virtual void computeRadiationProps(const ProcessorGroup* pc,
					 const Patch* patch,
					 CellInformation* cellinfo,
					 ArchesVariables* vars,
					 ArchesConstVariables* constvars);
      //
      /////////////////////////////////////////////////////////////////////////
      
      virtual void boundarycondition(const ProcessorGroup* pc,
                                      const Patch* patch,
      				      CellInformation* cellinfo, 
      				      ArchesVariables* vars,
      				      ArchesConstVariables* constvars);
      ////////////////////////////////////////////////////////////////////////

      virtual void intensitysolve(const ProcessorGroup* pc,
                                      const Patch* patch,
      				      CellInformation* cellinfo, 
      				      ArchesVariables* vars,
      				      ArchesConstVariables* constvars);
      ////////////////////////////////////////////////////////////////////////

protected: 
       // boundary condition
      BoundaryCondition* d_boundaryCondition;
private:

      double d_xumax;
      
      int d_sn, d_totalOrds; // totalOrdinates = sn*(sn+2)

      void computeOrdinatesOPL();

      int MAXITR;
      double QACCU, d_opl, af, qerr, totsrc;
      int iflag, iriter;
      int lambda;
      double wavemin, wavemax, dom, omega;
      int ffield;
      int wall;
      int symtry;
      int pfield;
      int sfield;
      int pbcfld;
      int outletfield;
      bool lprobone, lprobtwo, lprobthree, lradcal;

      OffsetArray1<double> fraction;

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





