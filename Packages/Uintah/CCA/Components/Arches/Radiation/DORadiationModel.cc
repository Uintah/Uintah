//----- DORadiationModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadLinearSolver.h>
//#include <Packages/Uintah/CCA/Components/Arches/Mixing/Common.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Core/Containers/OffsetArray1.h>
#include <Core/Thread/Time.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <math.h>
#include <Core/Math/MiscMath.h>


using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rordr_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rordrss_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rordrtn_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radcal_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdombc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomflux_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdombmcalc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomvolq_fort.h>
//****************************************************************************
// Default constructor for DORadiationModel
//****************************************************************************
DORadiationModel::DORadiationModel(BoundaryCondition* bndry_cond,
				   const ProcessorGroup* myworld):
                                   RadiationModel(),d_boundaryCondition(bndry_cond),
				   d_myworld(myworld)
{
  d_linearSolver = 0;
  /*
  rgamma = AllocMatrix(7,4);
  sd15 = AllocMatrix(80,6);
  sd = AllocMatrix(376,6);
  sd7 = AllocMatrix(16,3);
  sd3 = AllocMatrix(32,3);
  */
}

//****************************************************************************
// Destructor
//****************************************************************************
DORadiationModel::~DORadiationModel()
{
  delete d_linearSolver;
  /*
  DeallocMatrix(rgamma,4);
  DeallocMatrix(sd15,6);
  DeallocMatrix(sd,6);
  DeallocMatrix(sd7,3);
  DeallocMatrix(sd3,3);
  */
}

//****************************************************************************
// Problem Setup for DORadiationModel
//**************************************************************************** 

void 
DORadiationModel::problemSetup(const ProblemSpecP& params)

{
  ProblemSpecP db = params->findBlock("DORadiationModel");
  if (db) {
    db->getWithDefault("ordinates",d_sn,2);
//    db->getWithDefault("opl",d_xumax,3.0); too sensitive to have default
    db->require("opl",d_xumax);
  }
  else {
    d_sn=2;
    d_xumax=6.0;
  }
  lprobone = false;
  lprobtwo = false;
  lprobthree = false;
  lradcal = false;
  lambda = 1;

  fraction.resize(1,100);
  fraction.initialize(0.0);

  computeOrdinatesOPL();

  // ** WARNING ** ffield/Symmetry/sfield/outletfield hardcoded to -1,-3,-4,-5
  // These have been copied from BoundaryCondition.cc
  d_linearSolver = scinew RadLinearSolver(d_myworld);
  d_linearSolver->problemSetup(db);
  ffield = -1;
  symtry = -3;
  sfield = -4;
  outletfield = -5;
}

void
DORadiationModel::computeOrdinatesOPL() {

  d_opl = 0.6*d_xumax;
  if(lprobtwo==true){
    d_opl = 1.76;
  }
        d_totalOrds = d_sn*(d_sn+2);
//      d_totalOrds = 8*d_sn*d_sn;

  omu.resize(1,d_totalOrds + 1);
  oeta.resize(1,d_totalOrds + 1);
  oxi.resize(1,d_totalOrds + 1);
  wt.resize(1,d_totalOrds + 1);
  //  ord.resize(1,(d_sn/2) + 1);
  /*
   omu.resize(1,25);
   oeta.resize(1,25);
   oxi.resize(1,25);
   wt.resize(1,25);
  */
   //   ord.resize(1,3);

   omu.initialize(0.0);
   oeta.initialize(0.0);
   oxi.initialize(0.0);
   wt.initialize(0.0);
   //   ord.initialize(0.0);

                 fort_rordr(d_sn, oxi, omu, oeta, wt);
   //           fort_rordrss(d_sn, oxi, omu, oeta, wt);
   //           fort_rordrtn(d_sn, ord, oxi, omu, oeta, wt);
}

//****************************************************************************
//  Actually compute the properties here
//****************************************************************************

void 
DORadiationModel::computeRadiationProps(const ProcessorGroup*,
					 const Patch* patch,
					 CellInformation* cellinfo, 
					 ArchesVariables* vars,
					 ArchesConstVariables* constvars)

{
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    IntVector domLo = patch->getCellLowIndex();
    IntVector domHi = patch->getCellHighIndex();    

    fort_radcoef(idxLo, idxHi, constvars->temperature, 
		 constvars->co2, constvars->h2o, constvars->cellType, ffield, 
		 d_opl, constvars->sootFV, vars->ABSKG, vars->ESRCG,
		 cellinfo->xx, cellinfo->yy, cellinfo->zz, fraction, 
		 lprobone, lprobtwo, lprobthree, lambda, lradcal);
    /*
//<<<<<<< DORadiationModel.cc
//=======
    fort_radcoef(idxLo, idxHi, constvars->temperature, 
		 constvars->co2, constvars->h2o, constvars->cellType,
		 ffield, d_opl,
		 constvars->sootFV, vars->ABSKG, vars->ESRCG,
		 cellinfo->xx, cellinfo->yy, cellinfo->zz);
//>>>>>>> 1.3
    */
}

//***************************************************************************
// Sets the radiation boundary conditions for the D.O method
//***************************************************************************
void 
DORadiationModel::boundarycondition(const ProcessorGroup*,
					 const Patch* patch,
					 CellInformation* cellinfo,
					ArchesVariables* vars,
					ArchesConstVariables* constvars)
{
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    

  fort_rdombc(idxLo, idxHi, constvars->cellType, ffield, constvars->temperature,
	      vars->ABSKG,
	      xminus, xplus, yminus, yplus, zminus, zplus, lprobone, lprobtwo, lprobthree);

}
//***************************************************************************
// Solves for intensity in the D.O method
//***************************************************************************
void 
DORadiationModel::intensitysolve(const ProcessorGroup* pg,
					 const Patch* patch,
					 CellInformation* cellinfo,
					ArchesVariables* vars,
					ArchesConstVariables* constvars)
{
  double solve_start = Time::currentSeconds();
   rgamma.resize(1,29);
   sd15.resize(1,481);
   sd.resize(1,2257);
   sd7.resize(1,49);
   sd3.resize(1,97);

   rgamma.initialize(0.0);
   sd15.initialize(0.0);
   sd.initialize(0.0);
   sd7.initialize(0.0);
   sd3.initialize(0.0);

  wavemin = 50;
  wavemax = 10000;

  if (lambda > 1) {
  dom = (wavemax - wavemin)/(lambda - 1); 
  omega = wavemin - dom;
  fort_radarray(rgamma, sd15, sd, sd7, sd3);
  }

  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLo = patch->getCellLowIndex();
  IntVector domHi = patch->getCellHighIndex();
  
  int wall = d_boundaryCondition->wallCellType();
  double areaew;

  CCVariable<double> volume;
  CCVariable<double> su;
  CCVariable<double> aw;
  CCVariable<double> as;
  CCVariable<double> ab;
  CCVariable<double> ap;
  CCVariable<double> volq;
  CCVariable<double> cenint;
  
  vars->cenint.allocate(domLo,domHi);
  volume.allocate(domLo,domHi);
  su.allocate(domLo,domHi);
  aw.allocate(domLo,domHi);
  as.allocate(domLo,domHi);
  ab.allocate(domLo,domHi);
  ap.allocate(domLo,domHi);
  volq.allocate(domLo,domHi);
  arean.resize(domLo.x(),domHi.x());
  areatb.resize(domLo.x(),domHi.x());
  
  srcbm.resize(domLo.x(),domHi.x());
  srcbm.initialize(0.0);
  srcpone.resize(domLo.x(),domHi.x());
  srcpone.initialize(0.0);
  qfluxbbm.resize(domLo.x(),domHi.x());
  qfluxbbm.initialize(0.0);

  volume.initialize(0.0);
  volq.initialize(0.0);    
  arean.initialize(0.0);
  areatb.initialize(0.0);
  double timeRadMatrix = 0;
  double timeRadCoeffs = 0;
  vars->cenint.initialize(0.0);

  for (int bands =1; bands <=lambda; bands++)
  {
    volq.initialize(0.0);

  if(lradcal==true){    
              fort_radcal(idxLo, idxHi, vars->ABSKG, vars->ESRCG,
		cellinfo->xx, cellinfo->yy, cellinfo->zz, bands, dom, omega, constvars->cellType, ffield, constvars->co2, constvars->h2o, constvars->sootFV, constvars->temperature, lprobone, lprobtwo, lambda, fraction, rgamma, sd15, sd, sd7, sd3, d_opl);
  }

  for (int direcn = 1; direcn <=d_totalOrds; direcn++)
    {
      vars->cenint.initialize(0.0);
      su.initialize(0.0);
      aw.initialize(0.0);
      as.initialize(0.0);
      ab.initialize(0.0);
      ap.initialize(0.0);
      bool plusX, plusY, plusZ;
      fort_rdomsolve(idxLo, idxHi, constvars->cellType, wall, ffield, cellinfo->sew,
		     cellinfo->sns, cellinfo->stb, vars->ESRCG, direcn, oxi, omu,
		     oeta, wt, 
		     constvars->temperature, vars->ABSKG, vars->cenint, volume,
		     su, aw, as, ab, ap,
		     areaew, arean, areatb, volq, vars->src, plusX, plusY, plusZ, fraction, bands, vars->qfluxe, vars->qfluxw,
		    vars->qfluxn, vars->qfluxs,
		    vars->qfluxt, vars->qfluxb);
      //      double timeSetMat = Time::currentSeconds();
      d_linearSolver->setMatrix(pg ,patch, vars, plusX, plusY, 
				plusZ, su, ab, as, aw, ap);
      //      timeRadMatrix += Time::currentSeconds() - timeSetMat;
      bool converged =  d_linearSolver->radLinearSolve();
      if (converged) {
	d_linearSolver->copyRadSoln(patch, vars);
      }
      d_linearSolver->destroyMatrix();
      fort_rdomvolq(idxLo, idxHi, direcn, wt, vars->cenint, volq);
      fort_rdomflux(idxLo, idxHi, direcn, oxi, omu, oeta, wt, vars->cenint,
		    plusX, plusY, plusZ, vars->qfluxe, vars->qfluxw,
		    vars->qfluxn, vars->qfluxs,
		    vars->qfluxt, vars->qfluxb);
      }
      fort_rdomsrc(idxLo, idxHi, vars->ABSKG, vars->ESRCG,
		   volq, vars->src);
        }
  int me = d_myworld->myrank();
  if(me == 0) {
    cerr << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";
  }
  /*
   fort_rdombmcalc(idxLo, idxHi, cellinfo->xx, cellinfo->zz, srcbm, qfluxbbm, vars->src, vars->qfluxb, vars->qfluxw, lprobone, lprobtwo, lprobthree, srcpone, volq);
  */
}















