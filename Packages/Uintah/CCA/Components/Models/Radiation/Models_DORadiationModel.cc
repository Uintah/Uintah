//----- Models_DORadiationModel.cc --------------------------------------------------

#include <sci_defs/hypre_defs.h>

#include <fstream> // work around compiler bug with RHEL 3

#include <Core/Containers/OffsetArray1.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Time.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_CellInformationP.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_RadiationSolver.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_DORadiationModel.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_PetscSolver.h>
#ifdef HAVE_HYPRE
#include <Packages/Uintah/CCA/Components/Models/Radiation/Models_HypreSolver.h>
#endif
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <math.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rordr_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rordrss_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rordrtn_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_radarray_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_radcoef_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_radwsgg_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_radcal_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rdombc_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rdomsolve_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rdomsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rdomflux_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rdombmcalc_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rdomvolq_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rshsolve_fort.h>
#include <Packages/Uintah/CCA/Components/Models/Radiation/fortran/m_rshresults_fort.h>
//****************************************************************************
// Default constructor for Models_DORadiationModel
//****************************************************************************
Models_DORadiationModel::Models_DORadiationModel(const ProcessorGroup* myworld):
                                   Models_RadiationModel(),
				   d_myworld(myworld)
{
  d_linearSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
Models_DORadiationModel::~Models_DORadiationModel()
{
  delete d_linearSolver;

}

//****************************************************************************
// Problem Setup for Models_DORadiationModel
//**************************************************************************** 

void 
Models_DORadiationModel::problemSetup(const ProblemSpecP& params)

{
  ProblemSpecP db = params->findBlock("DORadiationModel");

  string prop_model;

  if (db) {
    db->getWithDefault("ordinates",d_sn,2);
    db->require("opl",d_opl);
    db->getWithDefault("property_model",prop_model,"radcoef");
    db->getWithDefault("spherical_harmonics",d_SHRadiationCalc,false);
  }
  else {
    d_sn=6;
    d_opl=0.18;
  }

  lprobone = false;
  lprobtwo = false;
  lprobthree = false;

  if (prop_model == "radcoef"){ 
    lradcal = false;
    lwsgg = false;
    lambda = 1;
    lplanckmean = false;
    lpatchmean = false;
  }

  if (prop_model == "patchmean"){ 
    lradcal = true;
    lwsgg = false;
    lambda = 6;
    lplanckmean = false;
    lpatchmean = true;
  }

  if (prop_model == "wsggm"){ 
    lradcal = false;
    lwsgg = true;
    lambda = 4;
    lplanckmean = false;
    lpatchmean = false;
  }

  fraction.resize(1,100);
  fraction.initialize(0.0);

  fractiontwo.resize(1,100);
  fractiontwo.initialize(0.0);

  computeOrdinatesOPL();

  string linear_sol;
  db->getWithDefault("linear_solver",linear_sol,"petsc");

  if (linear_sol == "petsc") d_linearSolver = scinew Models_PetscSolver(d_myworld);
#ifdef HAVE_HYPRE
  else if (linear_sol == "hypre") d_linearSolver = scinew Models_HypreSolver(d_myworld);
#endif
  
  d_linearSolver->problemSetup(db);

  // Variable to deal with cell typing used in radiation modules ...
  // may be useful later for people writing MPMICE code, when they 
  // want to deal with radiating walls. (first-order approximation).
  // Right now, I am hardcoding this to be -1; I will set all the pcell
  // values to the same thing for now.  When with Arches, this was 
  // defined in the BoundaryCondition class, but ICE does not have a 
  // boundary condition variable called pcell.
  //
  // Seshadri Kumar, April 11, 2005

  ffield = -1;

}

//**************************************************************************** 
// calculate the number of ordinates and their weights for the DO calculation
//**************************************************************************** 
void
Models_DORadiationModel::computeOrdinatesOPL() {

  d_totalOrds = d_sn*(d_sn+2);

  omu.resize(1,d_totalOrds + 1);
  oeta.resize(1,d_totalOrds + 1);
  oxi.resize(1,d_totalOrds + 1);
  wt.resize(1,d_totalOrds + 1);

  omu.initialize(0.0);
  oeta.initialize(0.0);
  oxi.initialize(0.0);
  wt.initialize(0.0);

  fort_m_rordr(d_sn, oxi, omu, oeta, wt);

  // The following are alternative ways of setting weights to the 
  // different ordinates ... I believe they should be options at the
  // problemSetup stage, but I have to talk with Gautham (SK, 04/08/05)
  // fort_m_rordrss(d_sn, oxi, omu, oeta, wt);
  // fort_m_rordrtn(d_sn, ord, oxi, omu, oeta, wt);

}


//****************************************************************************
//  Actually compute the properties here
//****************************************************************************

void 
Models_DORadiationModel::computeRadiationProps(const ProcessorGroup*,
					const Patch* patch,
					Models_CellInformation* cellinfo,
					RadiationVariables* vars,
					RadiationConstVariables* constvars)

{

  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLo = patch->getCellLowIndex();
  IntVector domHi = patch->getCellHighIndex();

  //  CCVariable<double> shgamma;
  //  vars->shgamma.allocate(domLo,domHi);
  //  vars->shgamma.initialize(0.0);
  
  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {

	IntVector currCell(ii,jj,kk);
	if (constvars->temperature[currCell] > 1000.0) {
	  double bc = constvars->mixfrac[currCell]*(84.0/100.0)*constvars->density[currCell];
	  double c3 = 0.1;
	  double rhosoot = 1950.0;
	  double cmw = 12.0;
	  double sootFactor = 0.01;

	  if (constvars->mixfrac[currCell] > 0.1)
	    vars->sootVF[currCell] = c3*bc*cmw/rhosoot*sootFactor;
	  else
	    vars->sootVF[currCell] = 0.0;
	}
      }
    }
  }

  fort_m_radcoef(idxLo, idxHi, 
	       vars->temperature, 
	       constvars->co2, 
	       constvars->h2o,
	       d_opl, 
	       vars->sootVF, 
	       vars->ABSKG, 
	       vars->ESRCG, 
	       vars->shgamma,
	       cellinfo->xx, 
	       cellinfo->yy, 
	       cellinfo->zz, 
	       fraction, 
	       fractiontwo,
	       lprobone, lprobtwo, lprobthree, 
	       lambda, 
	       lradcal);
}

//***************************************************************************
// Sets the radiation boundary conditions for the D.O method
//***************************************************************************
void 
Models_DORadiationModel::boundaryCondition(const ProcessorGroup*,
				    const Patch* patch,
				    RadiationVariables* vars)
{
    IntVector idxLo = patch->getCellFORTLowIndex();
    IntVector idxHi = patch->getCellFORTHighIndex();
    
    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    
    // temperature and ABSKG are modified in rdombc
    // cellType is a constVars variable in Arches, but here it is used
    // as ffield throughout the domain.  When MPMICE is implemented, it
    // may be useful to have a cellType variable.

    fort_m_rdombc(idxLo, idxHi, 
		vars->temperature,
		vars->ABSKG,
		xminus, xplus, yminus, yplus, zminus, zplus, 
		lprobone, lprobtwo, lprobthree);

}
//***************************************************************************
// Solves for intensity in the D.O method
//***************************************************************************
void 
Models_DORadiationModel::intensitysolve(const ProcessorGroup* pg,
				 const Patch* patch,
				 Models_CellInformation* cellinfo,
				 RadiationVariables* vars,
				 RadiationConstVariables* constvars)
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

  if (lambda > 1) {
    fort_m_radarray(rgamma, sd15, sd, sd7, sd3);
  }

  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  IntVector domLo = patch->getCellLowIndex();
  IntVector domHi = patch->getCellHighIndex();
  
  CCVariable<double> volume;
  volume.allocate(domLo,domHi);
  CCVariable<double> su;
  su.allocate(domLo,domHi);
  CCVariable<double> ae;
  ae.allocate(domLo,domHi);
  CCVariable<double> aw;
  aw.allocate(domLo,domHi);
  CCVariable<double> an;
  an.allocate(domLo,domHi);
  CCVariable<double> as;
  as.allocate(domLo,domHi);
  CCVariable<double> at;
  at.allocate(domLo,domHi);
  CCVariable<double> ab;
  ab.allocate(domLo,domHi);
  CCVariable<double> ap;
  ap.allocate(domLo,domHi);
  CCVariable<double> volq;
  volq.allocate(domLo,domHi);
  vars->cenint.allocate(domLo,domHi);
  CCVariable<int> cellType;
  cellType.allocate(domLo,domHi);

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
  //  double timeRadMatrix = 0;
  //  double timeRadCoeffs = 0;
  vars->cenint.initialize(0.0);

  //begin discrete ordinates
  if(d_SHRadiationCalc==false) {

    for (int bands =1; bands <=lambda; bands++) {

      volq.initialize(0.0);
      int ffield = -1;
      cellType.initialize(ffield);
      // I am retaining cellType here because we may need to have
      // cellType with MPMICE later, for the first-order radiation
      // effects    

      if(lwsgg == true){    
	fort_m_radwsgg(idxLo, idxHi, 
		     vars->ABSKG, 
		     vars->ESRCG, 
		     vars->shgamma,
		     bands, 
		     cellType, ffield, 
		     constvars->co2, 
		     constvars->h2o, 
		     constvars->sootVF, 
		     vars->temperature, 
		     lambda, 
		     fraction, 
		     fractiontwo);
      }

      if(lradcal==true){    
	fort_m_radcal(idxLo, idxHi, 
		    vars->ABSKG, 
		    vars->ESRCG, 
		    vars->shgamma,
		    cellinfo->xx, 
		    cellinfo->yy, 
		    cellinfo->zz, 
		    bands, 
		    cellType, ffield, 
		    constvars->co2, 
		    constvars->h2o, 
		    constvars->sootVF, 
		    vars->temperature, 
		    lprobone, lprobtwo, 
		    lplanckmean, lpatchmean, 
		    lambda, 
		    fraction, 
		    rgamma, sd15, sd, sd7, sd3, 
		    d_opl);
      }
      
      for (int direcn = 1; direcn <=d_totalOrds; direcn++) {
	vars->cenint.initialize(0.0);
	su.initialize(0.0);
	aw.initialize(0.0);
	as.initialize(0.0);
	ab.initialize(0.0);
	ap.initialize(0.0);
	ae.initialize(0.0);
	an.initialize(0.0);
	at.initialize(0.0);
	bool plusX, plusY, plusZ;

	fort_m_rdomsolve(idxLo, idxHi, 
		       cellType, 
		       ffield, 
		       cellinfo->sew,
		       cellinfo->sns, 
		       cellinfo->stb, 
		       vars->ESRCG, direcn, 
		       oxi, 
		       omu,
		       oeta, 
		       wt, 
		       vars->temperature, 
		       vars->ABSKG, 
		       vars->cenint, 
		       volume,
		       su, 
		       aw, as, ab, 
		       ap, 
		       ae, an, at,
		       volq, 
		       vars->src, 
		       plusX, plusY, plusZ, 
		       fraction, 
		       fractiontwo, 
		       bands, 
		       vars->qfluxe, 
		       vars->qfluxw,
		       vars->qfluxn, 
		       vars->qfluxs,
		       vars->qfluxt, 
		       vars->qfluxb, 
		       d_opl);
	
	//      double timeSetMat = Time::currentSeconds();

	d_linearSolver->setMatrix(pg, 
				  patch, 
				  vars, 
				  plusX, plusY, plusZ, 
				  su, 
				  ab, 
				  as, 
				  aw, 
				  ap, 
				  ae, 
				  an, 
				  at);
				
	//      timeRadMatrix += Time::currentSeconds() - timeSetMat;

	bool converged =  d_linearSolver->radLinearSolve();
	if (converged) {
	  d_linearSolver->copyRadSoln(patch, vars);
	}
	else {
	  if (pg->myrank() == 0) {
	    cerr << "radiation solver not converged" << endl;
	  }
	  exit(1);
	}
	d_linearSolver->destroyMatrix();

	fort_m_rdomvolq(idxLo, idxHi, 
		      direcn, 
		      wt, 
		      vars->cenint, volq);
	    
	fort_m_rdomflux(idxLo, idxHi, 
		      direcn, 
		      oxi, 
		      omu, 
		      oeta, 
		      wt, 
		      vars->cenint,
		      plusX, plusY, plusZ, 
		      vars->qfluxe, 
		      vars->qfluxw,
		      vars->qfluxn, 
		      vars->qfluxs,
		      vars->qfluxt, 
		      vars->qfluxb);
      }

      fort_m_rdomsrc(idxLo, idxHi, 
		   vars->ABSKG, 
		   vars->ESRCG,
		   volq, 
		   vars->src);
    }
    int me = d_myworld->myrank();
    if(me == 0) {
      cerr << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";
    }
    /*
      fort_m_rdombmcalc(idxLo, idxHi, cellType, ffield, cellinfo->xx, cellinfo->zz, cellinfo->sew, cellinfo->sns, cellinfo->stb, volume, areaew, arean, areatb, srcbm, qfluxbbm, vars->src, vars->qfluxe, vars->qfluxw, vars->qfluxn, vars->qfluxs, vars->qfluxt, vars->qfluxb, lprobone, lprobtwo, lprobthree, srcpone, volq, srcsum);
      
      cerr << "Total radiative source =" << srcsum << " watts\n";
    */
  }//end discrete ordinates


  if(d_SHRadiationCalc){

    double solve_start = Time::currentSeconds();
    for (int bands =1; bands <=lambda; bands++) {

      if(lwsgg == true){    
	fort_m_radwsgg(idxLo, idxHi, 
		     vars->ABSKG, 
		     vars->ESRCG, 
		     vars->shgamma,
		     bands, 
		     cellType, ffield, 
		     constvars->co2, 
		     constvars->h2o, 
		     constvars->sootVF, 
		     vars->temperature, 
		     lambda, 
		     fraction, 
		     fractiontwo);
      }

      if(lradcal==true){    
	fort_m_radcal(idxLo, idxHi, 
		    vars->ABSKG, 
		    vars->ESRCG, 
		    vars->shgamma,
		    cellinfo->xx, 
		    cellinfo->yy, 
		    cellinfo->zz, 
		    bands, 
		    cellType, ffield, 
		    constvars->co2, 
		    constvars->h2o, 
		    constvars->sootVF, 
		    vars->temperature, 
		    lprobone, lprobtwo, 
		    lplanckmean, lpatchmean, 
		    lambda, 
		    fraction, 
		    rgamma, sd15, 
		    sd, sd7, 
		    sd3, 
		    d_opl);
      }

      vars->cenint.initialize(0.0);
      su.initialize(0.0);
      ae.initialize(0.0);
      aw.initialize(0.0);
      an.initialize(0.0);
      as.initialize(0.0);
      at.initialize(0.0);
      ab.initialize(0.0);
      ap.initialize(0.0);
      bool plusX, plusY, plusZ;

      fort_m_rshsolve(idxLo, idxHi, 
		    cellType, ffield, 
		    cellinfo->sew, 
		    cellinfo->sns, 
		    cellinfo->stb,
		    cellinfo->xx, 
		    cellinfo->yy, 
		    cellinfo->zz,
		    vars->ESRCG, 
		    vars->temperature, 
		    vars->ABSKG, 
		    vars->shgamma, 
		    volume, 
		    su, 
		    ae, 
		    aw, 
		    an, 
		    as, 
		    at, 
		    ab, 
		    ap,
		    volq, 
		    vars->src, 
		    plusX, plusY, plusZ, 
		    fraction, 
		    fractiontwo, 
		    bands);

      //      double timeSetMat = Time::currentSeconds();
      d_linearSolver->setMatrix(pg ,
				patch, 
				vars, 
				plusX, plusY, plusZ, 
				su, 
				ab, 
				as, 
				aw, 
				ap, 
				ae, 
				an, 
				at);
				
      //      timeRadMatrix += Time::currentSeconds() - timeSetMat;

      bool converged =  d_linearSolver->radLinearSolve();
      if (converged) {
	d_linearSolver->copyRadSoln(patch, vars);
      }
      else {
        if (pg->myrank() == 0) {
          cerr << "radiation solver not converged" << endl;
        }
        exit(1);
      }
      d_linearSolver->destroyMatrix();

      fort_m_rshresults(idxLo, idxHi, 
		      vars->cenint, 
		      volq,
		      cellType, ffield,
		      cellinfo->xx, 
		      cellinfo->yy, 
		      cellinfo->zz,
		      vars->temperature,
		      vars->qfluxe, 
		      vars->qfluxw,
		      vars->qfluxn, 
		      vars->qfluxs,
		      vars->qfluxt, 
		      vars->qfluxb,
		      vars->ABSKG, 
		      vars->shgamma, 
		      vars->ESRCG, 
		      vars->src, 
		      fraction, 
		      fractiontwo, 
		      bands);
    }

    int me = d_myworld->myrank();
    if(me == 0) {
      cerr << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";
    }
    /*
      fort_m_rdombmcalc(idxLo, idxHi, cellType, ffield, cellinfo->xx, cellinfo->zz, cellinfo->sew, cellinfo->sns, cellinfo->stb, volume, areaew, arean, areatb, srcbm, qfluxbbm, vars->src, vars->qfluxe, vars->qfluxw, vars->qfluxn, vars->qfluxs, vars->qfluxt, vars->qfluxb, lprobone, lprobtwo, lprobthree, srcpone, volq, srcsum);
      
      cerr << "Total radiative source =" << srcsum << " watts\n";
    */
  }
}
