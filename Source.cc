//----- Source.cc ----------------------------------------------

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SoleVariable.h>

using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/mascal_scalar_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mascal_fort.h>
#ifdef divergenceconstraint
#include <Packages/Uintah/CCA/Components/Arches/fortran/pressrcpred_var_fort.h>
#else
#include <Packages/Uintah/CCA/Components/Arches/fortran/pressrcpred_fort.h>
#endif
#include <Packages/Uintah/CCA/Components/Arches/fortran/add_mm_enth_src_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradthinsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmmomsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/scalsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/uvelsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/vvelsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/wvelsrc_fort.h>

//****************************************************************************
// Constructor for Source
//****************************************************************************
Source::Source(PhysicalConstants* phys_const)
                           :d_physicalConsts(phys_const)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
Source::~Source()
{
}

//****************************************************************************
// Problem Setup
//****************************************************************************
void 
Source::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP params_non_constant = params;
  const ProblemSpecP params_root = params_non_constant->getRootNode();
  ProblemSpecP db=params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("MMS");

  db->getWithDefault("whichMMS", d_mms, "constantMMS");
  if (d_mms == "constantMMS"){
    ProblemSpecP db_mms = db->findBlock("constantMMS");
    db_mms->getWithDefault("cu",cu,1.0);
    db_mms->getWithDefault("cv",cv,1.0);
    db_mms->getWithDefault("cw",cw,1.0);
    db_mms->getWithDefault("cp",cp,1.0);
    db_mms->getWithDefault("phi0",phi0,0.5);
  }
  else if (d_mms == "gao1MMS") {
    ProblemSpecP db_mms = db->findBlock("gao1MMS");
    db_mms->require("rhoair", d_airDensity);
    db_mms->require("rhohe", d_heDensity);
    db_mms->require("gravity", d_gravity);//Vector
    db_mms->require("viscosity",d_viscosity); 
    db_mms->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
    db_mms->getWithDefault("cu",cu,1.0);
    db_mms->getWithDefault("cv",cv,1.0);
    db_mms->getWithDefault("cw",cw,1.0);
    db_mms->getWithDefault("cp",cp,1.0);
    db_mms->getWithDefault("phi0",phi0,0.5);
  }
  else if (d_mms == "thornock1MMS") {
	  ProblemSpecP db_mms = db->findBlock("thornock1MMS");
	  db_mms->require("cu",cu);
  }
  else if (d_mms == "almgrenMMS") {
	  ProblemSpecP db_mms = db->findBlock("almgrenMMS");
	  db_mms->require("amplitude",amp);
  }
  else
	  throw InvalidValue("current MMS "
		       "not supported: " + d_mms, __FILE__, __LINE__);
}
 

//****************************************************************************
// Velocity source calculation
//****************************************************************************
void 
Source::calculateVelocitySource(const ProcessorGroup* pc ,
				const Patch* patch,
				double delta_t,
				int index,
				CellInformation* cellinfo,
				ArchesVariables* vars,
				ArchesConstVariables* constvars)
{
  
  //get index component of gravity
  double gravity = d_physicalConsts->getGravity(index);
  // Get the patch and variable indices
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  
  switch(index) {
  case Arches::XDIR:
{
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_uvelsrc(idxLoU, idxHiU, constvars->uVelocity, constvars->old_uVelocity,
		 vars->uVelNonlinearSrc, vars->uVelLinearSrc,
		 constvars->vVelocity, constvars->wVelocity, constvars->density,
		 constvars->viscosity, constvars->old_density,
		 constvars->denRefArray,
		 gravity, delta_t,  cellinfo->ceeu, cellinfo->cweu, 
		 cellinfo->cwwu, cellinfo->cnn, cellinfo->csn, cellinfo->css,
		 cellinfo->ctt, cellinfo->cbt, cellinfo->cbb, cellinfo->sewu,
		 cellinfo->sew, cellinfo->sns, cellinfo->stb, cellinfo->dxpw,
		 cellinfo->fac1u, cellinfo->fac2u, cellinfo->fac3u,
		 cellinfo->fac4u, cellinfo->iesdu, cellinfo->iwsdu);

// ++ jeremy ++
	if (d_boundaryCondition->getNumSourceBndry() > 0){	
		for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
			vars->uVelNonlinearSrc[*iter] += vars->umomBoundarySrc[*iter];
		}
	}
	
// -- jeremy -- 
}
    break;
  case Arches::YDIR:
  {	
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_vvelsrc(idxLoV, idxHiV, constvars->vVelocity, constvars->old_vVelocity,
		 vars->vVelNonlinearSrc, vars->vVelLinearSrc,
		 constvars->uVelocity, constvars->wVelocity, constvars->density,
		 constvars->viscosity, constvars->old_density,
		 constvars->denRefArray,
		 gravity, delta_t,
		 cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		 cellinfo->cnnv, cellinfo->csnv, cellinfo->cssv,
		 cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		 cellinfo->sew, cellinfo->snsv, cellinfo->sns, cellinfo->stb,
		 cellinfo->dyps, cellinfo->fac1v, cellinfo->fac2v,
		 cellinfo->fac3v, cellinfo->fac4v, cellinfo->jnsdv,
		 cellinfo->jssdv); 

// ++ jeremy ++
	if (d_boundaryCondition->getNumSourceBndry() > 0){	
		for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
			vars->vVelNonlinearSrc[*iter] += vars->vmomBoundarySrc[*iter];
		}
	}
//-- jeremy -- 	
}
    break;
  case Arches::ZDIR:
{
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_wvelsrc(idxLoW, idxHiW, constvars->wVelocity, constvars->old_wVelocity,
		 vars->wVelNonlinearSrc, vars->wVelLinearSrc,
		 constvars->uVelocity, constvars->vVelocity, constvars->density,
		 constvars->viscosity, constvars->old_density,
		 constvars->denRefArray,
		 gravity, delta_t,
		 cellinfo->cee, cellinfo->cwe, cellinfo->cww,
		 cellinfo->cnn, cellinfo->csn, cellinfo->css,
		 cellinfo->cttw, cellinfo->cbtw, cellinfo->cbbw,
		 cellinfo->sew, cellinfo->sns, cellinfo->stbw,
		 cellinfo->stb, cellinfo->dzpb, cellinfo->fac1w,
		 cellinfo->fac2w, cellinfo->fac3w, cellinfo->fac4w,
		 cellinfo->ktsdw, cellinfo->kbsdw); 

// ++ jeremy ++ 
	if (d_boundaryCondition->getNumSourceBndry() > 0){	
		for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
			vars->wVelNonlinearSrc[*iter] += vars->wmomBoundarySrc[*iter];
		}
	}
// -- jeremy -- 	

}
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcVelSrc", __FILE__, __LINE__);
  }


}

//****************************************************************************
// Pressure source calculation
//****************************************************************************
void 
Source::calculatePressureSourcePred(const ProcessorGroup* ,
				    const Patch* patch,
				    double delta_t,
				    CellInformation* cellinfo,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars,
                                    bool doing_EKT_now)
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  if (!(doing_EKT_now)) {
#ifdef divergenceconstraint
  fort_pressrcpred_var(idxLo, idxHi, vars->pressNonlinearSrc,
		       constvars->divergence, constvars->uVelRhoHat,
		       constvars->vVelRhoHat, constvars->wVelRhoHat, delta_t,
		       cellinfo->sew, cellinfo->sns, cellinfo->stb);
#else
  fort_pressrcpred(idxLo, idxHi, vars->pressNonlinearSrc,
		   constvars->density, constvars->uVelRhoHat,
                   constvars->vVelRhoHat, constvars->wVelRhoHat, delta_t,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb);
#endif
  }
  for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	IntVector currcell(ii,jj,kk);
	vars->pressNonlinearSrc[currcell] -= constvars->filterdrhodt[currcell]/delta_t;
      }
    }
  }
}


//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::calculateScalarSource(const ProcessorGroup* pc,
			      const Patch* patch,
			      double delta_t,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getExtraCellLowIndex__New(), patch->getExtraCellHighIndex__New());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  fort_scalsrc(idxLo, idxHi, vars->scalarLinearSrc, vars->scalarNonlinearSrc,
	       constvars->old_density, constvars->old_scalar,
	       cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);

// Here we need to add the boundary source term if there are some.
	if (d_boundaryCondition->getNumSourceBndry() > 0){
		for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
			vars->scalarNonlinearSrc[*iter] += vars->scalarBoundarySrc[*iter];
		}
	}
}

//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::calculateExtraScalarSource(const ProcessorGroup* pc,
			      const Patch* patch,
			      double delta_t,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getExtraCellLowIndex__New(), patch->getExtraCellHighIndex__New());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  fort_scalsrc(idxLo, idxHi, vars->scalarLinearSrc, vars->scalarNonlinearSrc,
	       constvars->old_density, constvars->old_scalar,
	       cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);
}


//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::addReactiveScalarSource(const ProcessorGroup*,
				const Patch* patch,
				double,
				CellInformation* cellinfo,
				ArchesVariables* vars,
				ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector indexLow = patch->getFortranCellLowIndex__New();
  IntVector indexHigh = patch->getFortranCellHighIndex__New();
  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	vars->scalarNonlinearSrc[currCell] += vol*
					constvars->reactscalarSRC[currCell]*
	                                constvars->density[currCell];
      }
    }
  }
}

//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::calculateEnthalpySource(const ProcessorGroup*,
			      const Patch* patch,
			      double delta_t,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getExtraCellLowIndex__New(), patch->getExtraCellHighIndex__New());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  fort_scalsrc(idxLo, idxHi, vars->scalarLinearSrc, vars->scalarNonlinearSrc,
	       constvars->old_density, constvars->old_enthalpy,
	       cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);

// ++ jeremy ++
	if (d_boundaryCondition->getNumSourceBndry() > 0) {
		for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
			vars->scalarNonlinearSrc[*iter] += vars->enthalpyBoundarySrc[*iter];
		}
	}
// -- jeremy -- 


}


//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::computeEnthalpyRadThinSrc(const ProcessorGroup*,
				  const Patch* patch,
				  CellInformation* cellinfo,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  double tref = 298; // warning, read it in
  fort_enthalpyradthinsrc(idxLo, idxHi, vars->scalarNonlinearSrc,
			  vars->temperature, constvars->absorption,
			  cellinfo->sew, cellinfo->sns, cellinfo->stb, tref);
}

//****************************************************************************
// Calls Fortran MASCAL
//****************************************************************************
void 
Source::modifyVelMassSource(const ProcessorGroup* ,
			    const Patch* patch,
			    double,
			    int index,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{
  // Get the patch and variable indices
  // And call the fortran routine (MASCAL)
  IntVector idxLo;
  IntVector idxHi;
  IntVector domLo;
  IntVector domHi;
  switch(index) {
  case Arches::XDIR:
    idxLo = patch->getSFCXFORTLowIndex();
    idxHi = patch->getSFCXFORTHighIndex();
    fort_mascal(idxLo, idxHi, constvars->uVelocity,
		vars->uVelocityCoeff[Arches::AE],
		vars->uVelocityCoeff[Arches::AW],
		vars->uVelocityCoeff[Arches::AN],
		vars->uVelocityCoeff[Arches::AS],
		vars->uVelocityCoeff[Arches::AT],
		vars->uVelocityCoeff[Arches::AB],
		vars->uVelNonlinearSrc,
		vars->uVelocityConvectCoeff[Arches::AE],
		vars->uVelocityConvectCoeff[Arches::AW],
		vars->uVelocityConvectCoeff[Arches::AN],
		vars->uVelocityConvectCoeff[Arches::AS],
		vars->uVelocityConvectCoeff[Arches::AT],
		vars->uVelocityConvectCoeff[Arches::AB]);

    break;
  case Arches::YDIR:
    idxLo = patch->getSFCYFORTLowIndex();
    idxHi = patch->getSFCYFORTHighIndex();
    fort_mascal(idxLo, idxHi, constvars->vVelocity,
		vars->vVelocityCoeff[Arches::AE],
		vars->vVelocityCoeff[Arches::AW],
		vars->vVelocityCoeff[Arches::AN],
		vars->vVelocityCoeff[Arches::AS],
		vars->vVelocityCoeff[Arches::AT],
		vars->vVelocityCoeff[Arches::AB],
		vars->vVelNonlinearSrc,
		vars->vVelocityConvectCoeff[Arches::AE],
		vars->vVelocityConvectCoeff[Arches::AW],
		vars->vVelocityConvectCoeff[Arches::AN],
		vars->vVelocityConvectCoeff[Arches::AS],
		vars->vVelocityConvectCoeff[Arches::AT],
		vars->vVelocityConvectCoeff[Arches::AB]);

    break;
  case Arches::ZDIR:
    idxLo = patch->getSFCZFORTLowIndex();
    idxHi = patch->getSFCZFORTHighIndex();
    fort_mascal(idxLo, idxHi, constvars->wVelocity,
		vars->wVelocityCoeff[Arches::AE],
		vars->wVelocityCoeff[Arches::AW],
		vars->wVelocityCoeff[Arches::AN],
		vars->wVelocityCoeff[Arches::AS],
		vars->wVelocityCoeff[Arches::AT],
		vars->wVelocityCoeff[Arches::AB],
		vars->wVelNonlinearSrc,
		vars->wVelocityConvectCoeff[Arches::AE],
		vars->wVelocityConvectCoeff[Arches::AW],
		vars->wVelocityConvectCoeff[Arches::AN],
		vars->wVelocityConvectCoeff[Arches::AS],
		vars->wVelocityConvectCoeff[Arches::AT],
		vars->wVelocityConvectCoeff[Arches::AB]);

    break;
  default:
    throw InvalidValue("Invalid index in Source::calcVelMassSrc", __FILE__, __LINE__);
  }
}


//****************************************************************************
// Documentation here
// FORT_MASCAL
//****************************************************************************
void 
Source::modifyScalarMassSource(const ProcessorGroup* ,
			       const Patch* patch,
			       double,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars,
			       int conv_scheme)
{
  // Get the patch and variable indices
  // And call the fortran routine (MASCAL)
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  fort_mascalscalar(idxLo, idxHi, constvars->scalar,
		    vars->scalarCoeff[Arches::AE],
		    vars->scalarCoeff[Arches::AW],
		    vars->scalarCoeff[Arches::AN],
		    vars->scalarCoeff[Arches::AS],
		    vars->scalarCoeff[Arches::AT],
		    vars->scalarCoeff[Arches::AB],
		    vars->scalarNonlinearSrc,
		    vars->scalarConvectCoeff[Arches::AE],
		    vars->scalarConvectCoeff[Arches::AW],
		    vars->scalarConvectCoeff[Arches::AN],
		    vars->scalarConvectCoeff[Arches::AS],
		    vars->scalarConvectCoeff[Arches::AT],
		    vars->scalarConvectCoeff[Arches::AB],
		    conv_scheme);
}

void 
Source::modifyEnthalpyMassSource(const ProcessorGroup* ,
			       const Patch* patch,
			       double,
			       ArchesVariables* vars,
			       ArchesConstVariables* constvars,
			       int conv_scheme)
{
  // Get the patch and variable indices
  // And call the fortran routine (MASCAL)
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  fort_mascalscalar(idxLo, idxHi, constvars->enthalpy,
		    vars->scalarCoeff[Arches::AE],
		    vars->scalarCoeff[Arches::AW],
		    vars->scalarCoeff[Arches::AN],
		    vars->scalarCoeff[Arches::AS],
		    vars->scalarCoeff[Arches::AT],
		    vars->scalarCoeff[Arches::AB],
		    vars->scalarNonlinearSrc,
		    vars->scalarConvectCoeff[Arches::AE],
		    vars->scalarConvectCoeff[Arches::AW],
		    vars->scalarConvectCoeff[Arches::AN],
		    vars->scalarConvectCoeff[Arches::AS],
		    vars->scalarConvectCoeff[Arches::AT],
		    vars->scalarConvectCoeff[Arches::AB],
		    conv_scheme);
}

//****************************************************************************
// Add the momentum source from continuous solid-gas momentum exchange
//****************************************************************************

void 
Source::computemmMomentumSource(const ProcessorGroup*,
				const Patch* patch,
				int index,
				CellInformation*,
				ArchesVariables* vars,
				ArchesConstVariables* constvars)
{
  IntVector idxLoU;
  IntVector idxHiU;
  
  switch(index) {
  case 1:
  // Get the low and high index for the patch and the variables
  idxLoU = patch->getSFCXFORTLowIndex();
  idxHiU = patch->getSFCXFORTHighIndex();
  

  fort_mmmomsrc(idxLoU, idxHiU, vars->uVelNonlinearSrc, vars->uVelLinearSrc,
		constvars->mmuVelSu, constvars->mmuVelSp);
  break;
  case 2:
  // Get the low and high index for the patch and the variables
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();

    fort_mmmomsrc(idxLoU, idxHiU, vars->vVelNonlinearSrc, vars->vVelLinearSrc,
		  constvars->mmvVelSu, constvars->mmvVelSp);
    break;
  case 3:
  // Get the low and high index for the patch and the variables
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
  
    fort_mmmomsrc(idxLoU, idxHiU, vars->wVelNonlinearSrc, vars->wVelLinearSrc,
		  constvars->mmwVelSu, constvars->mmwVelSp);
    break;
  default:
    cerr << "Invalid Index value" << endl;
    break;
  }

  // add in su and sp terms from multimaterial based on index
}

//****************************************************************************
// Add the enthalpy source from continuous solid-gas energy exchange
//****************************************************************************

void 
Source::addMMEnthalpySource(const ProcessorGroup* ,
			    const Patch* patch,
			    CellInformation* ,
			    ArchesVariables* vars,
			    ArchesConstVariables* constvars)
{

  // Get the low and high index for the patch

  IntVector valid_lo = patch->getFortranCellLowIndex__New();
  IntVector valid_hi = patch->getFortranCellHighIndex__New();

  fort_add_mm_enth_src(vars->scalarNonlinearSrc,
		       vars->scalarLinearSrc,
		       constvars->mmEnthSu,
		       constvars->mmEnthSp,
		       valid_lo,
		       valid_hi);


}

//****************************************************************************
// Velocity source calculation for MMS
//****************************************************************************
void 
Source::calculateVelMMSSource(const ProcessorGroup* ,
				const Patch* patch,
				double delta_t, double time,
				int index,
				CellInformation* cellinfo,
				ArchesVariables* vars,
				ArchesConstVariables* constvars)
{
//  double time = d_lab->d_sharedState->getElapsedTime();
  // Get the patch and variable indices
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();

  double rho0=0.0;
  switch(index) {
  case Arches::XDIR:

  for (int colZ = idxLoU.z(); colZ <= idxHiU.z(); colZ ++) {
    for (int colY = idxLoU.y(); colY <= idxHiU.y(); colY ++) {
      for (int colX = idxLoU.x(); colX <= idxHiU.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);

	//Make sure that this is the density you want
	rho0 = constvars->new_density[currCell];

	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "gao1MMS") {
		vars->uVelNonlinearSrc[currCell] += rho0*(2.0*cu*cu+cu*cv+cu*cw)*cellinfo->xu[colX]
			+rho0*(2.0*cu+cv+cw)*time+rho0+cp-(rho0-d_airDensity)*d_gravity.x();
	}
      }
    }
  }


    break;
  case Arches::YDIR:
    
  for (int colZ = idxLoV.z(); colZ <= idxHiV.z(); colZ ++) {
    for (int colY = idxLoV.y(); colY <= idxHiV.y(); colY ++) {
      for (int colX = idxLoV.x(); colX <= idxHiV.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);

	//This density should change depending on what you are verifying...
	rho0 = constvars->new_density[currCell];

	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "gao1MMS") {
		vars->vVelNonlinearSrc[currCell] +=  rho0*(2.0*cv*cv+cu*cv+cv*cw)*cellinfo->yv[colY]
                	+rho0*(2.0*cv+cu+cw)*time+rho0+cp-(rho0-d_airDensity)*d_gravity.y();
	}
      }
    }
  }


    break;
  case Arches::ZDIR:

  for (int colZ = idxLoW.z(); colZ <= idxHiW.z(); colZ ++) {
    for (int colY = idxLoW.y(); colY <= idxHiW.y(); colY ++) {
      for (int colX = idxLoW.x(); colX <= idxHiW.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	//This density should change depending on what you are verifying...
	rho0 = constvars->new_density[currCell];
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "gao1MMS") {
		vars->wVelNonlinearSrc[currCell] +=  rho0*(2.0*cw*cw+cu*cw+cv*cw)*cellinfo->zw[colX]
                	+rho0*(2.0*cw+cu+cv)*time+rho0+cp-(rho0-d_airDensity)*d_gravity.z();
	}
      }
    }
  }


    break;
  default:
    throw InvalidValue("Invalid index in LinearSource::calcVelSrc", __FILE__, __LINE__);
  }

}
//****************************************************************************
// Scalar source calculation for MMS
//****************************************************************************
void 
Source::calculateScalarMMSSource(const ProcessorGroup*,
			      const Patch* patch,
			      double delta_t,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  double rho0=0.0;
  
 for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "gao1MMS") {
	  rho0 = constvars->density[currCell];
	  vars->scalarNonlinearSrc[currCell] += rho0
                                                *phi0*(cu+cv+cw);
	}
      }
    }
  }

}
//****************************************************************************
// Pressure source calculation for MMS
//****************************************************************************
void 
Source::calculatePressMMSSourcePred(const ProcessorGroup* ,
				    const Patch* patch,
				    double delta_t,
				    CellInformation* cellinfo,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars)
{

  // Get the patch and variable indices
  double rho0 = 0.0;
  IntVector idxLo = patch->getFortranCellLowIndex__New();
  IntVector idxHi = patch->getFortranCellHighIndex__New();
  
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	rho0 = constvars->density[currCell];
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "gao1MMS") {
		vars->pressNonlinearSrc[currCell] += rho0*(cu+cv+cw);
	}
      }
    }
  }
}

