//----- DORadiationModel.cc --------------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadLinearSolver.h>
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
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/radcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomr_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdombc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/fortran/rdomflux_fort.h>
//****************************************************************************
// Default constructor for DORadiationModel
//****************************************************************************
DORadiationModel::DORadiationModel(BoundaryCondition* bndry_cond,
				   const ProcessorGroup* myworld):
                                   RadiationModel(),d_boundaryCondition(bndry_cond),
				   d_myworld(myworld)
{
  d_linearSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
DORadiationModel::~DORadiationModel()
{
  delete d_linearSolver;
}

//****************************************************************************
// Problem Setup for DORadiationModel
//**************************************************************************** 

void 
DORadiationModel::problemSetup(const ProblemSpecP& params)

{
  ProblemSpecP db = params->findBlock("DORadiationModel");
  if (db) {
    if (db->findBlock("ordinates"))
      db->require("ordinates",d_sn);
    else
      d_sn = 2;
    if (db->findBlock("opl"))
      db->require("opl",d_xumax);
    else
      d_xumax = 3.0;
  }
  else {
    d_sn = 2;
    d_xumax=3.0;
  }
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
  d_totalOrds = d_sn*(d_sn+2);

  //   omu.resize(1,d_totalOrds + 1);
  //   oeta.resize(1,d_totalOrds + 1);
  //   oxi.resize(1,d_totalOrds + 1);
  //   wt.resize(1,d_totalOrds + 1);
  //   ord.resize(1,d_sn/2 + 1);

   omu.resize(1,90);
   oeta.resize(1,90);
   oxi.resize(1,90);
   wt.resize(1,90);
   ord.resize(1,6);

   omu.initialize(0.0);
   oeta.initialize(0.0);
   oxi.initialize(0.0);
   wt.initialize(0.0);
   ord.initialize(0.0);
 
  fort_rordr(d_sn, ord, oxi, omu, oeta, wt);
}

//****************************************************************************
// Radiation Initializations
//****************************************************************************
void 
DORadiationModel::radiationInitialize()
{
  MAXITR = 1;
  QACCU  = 0.0001;
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
		 constvars->co2, constvars->h2o, constvars->cellType,
		 ffield, d_opl,
		 constvars->sootFV, vars->ABSKG, vars->ESRCG,
		 cellinfo->xx, cellinfo->yy, cellinfo->zz);
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
	      xminus, xplus, yminus, yplus, zminus, zplus);
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
  
  volume.initialize(0.0);
  volq.initialize(0.0);    
  arean.initialize(0.0);
  areatb.initialize(0.0);
  double timeRadMatrix = 0;
  double timeRadCoeffs = 0;
  vars->cenint.initialize(0.0);
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
		     areaew, arean, areatb, volq, vars->src, plusX, plusY, plusZ);
      //      double timeSetMat = Time::currentSeconds();
      d_linearSolver->setMatrix(pg ,patch, vars, plusX, plusY, 
				plusZ, su, ab, as, aw, ap);
      //      timeRadMatrix += Time::currentSeconds() - timeSetMat;
      bool converged =  d_linearSolver->radLinearSolve();
      if (converged) {
	d_linearSolver->copyRadSoln(patch, vars);
      }
      d_linearSolver->destroyMatrix();
      fort_rdomsrc(idxLo, idxHi, direcn, wt, vars->ABSKG, vars->ESRCG,
		   vars->cenint, volq, vars->src);
      fort_rdomflux(idxLo, idxHi, direcn, oxi, omu, oeta, wt, vars->cenint,
		    plusX, plusY, plusZ, vars->qfluxe, vars->qfluxw,
		    vars->qfluxn, vars->qfluxs,
		    vars->qfluxt, vars->qfluxb);
      }
  int me = d_myworld->myrank();
  if(me == 0) {
    cerr << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";
  }

}


//***************************************************************************
// Intensity Iterations and computing heat flux divergence
//***************************************************************************
#if 0
void 
DORadiationModel::computeHeatFluxDiv(const ProcessorGroup*,
					 const Patch* patch,
					 CellInformation* cellinfo,
					ArchesVariables* vars)

{
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();

    int wall = d_boundaryCondition->wallCellType();
    //    int pbcfld = d_pressureBdry->d_cellTypeID;

    idxHi = idxHi + IntVector(2,2,2);
    idxLo = idxLo - IntVector(1,1,1);

    CCVariable<double> xintbc;
    CCVariable<double> xintfc;
    CCVariable<double> yintbc;
    CCVariable<double> yintfc;
    CCVariable<double> zintbc;
    CCVariable<double> zintfc;
    CCVariable<double> cintm;
    CCVariable<double> volq;
    CCVariable<double> qince;
    CCVariable<double> qincw;
    CCVariable<double> qincn;
    CCVariable<double> qincs;
    CCVariable<double> qinct;
    CCVariable<double> qincb;

    xintbc.allocate(idxLo,idxHi);
    xintfc.allocate(idxLo,idxHi);
    yintbc.allocate(idxLo,idxHi);
    yintfc.allocate(idxLo,idxHi);
    zintbc.allocate(idxLo,idxHi);
    zintfc.allocate(idxLo,idxHi);
     cintm.allocate(idxLo,idxHi);
      volq.allocate(idxLo,idxHi);
     qince.allocate(idxLo,idxHi);
     qincw.allocate(idxLo,idxHi);
     qincn.allocate(idxLo,idxHi);
     qincs.allocate(idxLo,idxHi);
     qinct.allocate(idxLo,idxHi);
     qincb.allocate(idxLo,idxHi);

    idxHi = idxHi - IntVector(2,2,2);
    idxLo = idxLo + IntVector(1,1,1);

    xintbc.initialize(0.0);
    xintfc.initialize(0.0);
    yintbc.initialize(0.0);
    yintfc.initialize(0.0);
    zintbc.initialize(0.0);
    zintfc.initialize(0.0);
     cintm.initialize(0.0);
      volq.initialize(0.0);
     qince.initialize(0.0);
     qincw.initialize(0.0);
     qincn.initialize(0.0);
     qincs.initialize(0.0);
     qinct.initialize(0.0);
     qincb.initialize(0.0);

    fort_rdomr(idxLo, idxHi, d_sn, oxi, omu, oeta, wt, 
	       vars->temperature, cellinfo->xu, cellinfo->yv, cellinfo->zw, 
	       vars->cellType,  cellinfo->sew, cellinfo->sns, cellinfo->stb,
	       xintbc, xintfc, yintbc, yintfc, zintbc, zintfc, cintm, 
	       vars->ABSKG, 
	       vars->ESRCG, volq, vars->qfluxe, vars->qfluxw,
	       vars->qfluxn, vars->qfluxs, vars->qfluxt, vars->qfluxb,
	       qince, qincw, qincn, qincs, qinct, qincb, vars->src, 
	       wall, symtry, ffield,
	       pfield, sfield, pbcfld, d_opl, MAXITR, QACCU, totsrc,  
	       iflag, iriter, af, qerr);
}

#endif












