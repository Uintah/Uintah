/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
#include <CCA/Components/Arches/Radiation/RadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadiationSolver.h>

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
                                         ArchesConstVariables* constvars,
                                         bool d_abskp);
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





