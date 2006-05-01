//----- Source.cc ----------------------------------------------

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/StencilMatrix.h>
#include <Packages/Uintah/CCA/Components/Arches/TurbulenceModel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
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
#include <Packages/Uintah/CCA/Components/Arches/fortran/computeVel_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradflux_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradthinsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/add_mm_enth_src_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/addpressgrad_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/addtranssrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/calcpressgrad_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradflux_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradthinsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/mmmomsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/scalsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/uvelsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/vvelsrc_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/wvelsrc_fort.h>

//****************************************************************************
// Constructor for Source
//****************************************************************************
Source::Source(TurbulenceModel* turb_model, PhysicalConstants* phys_const)
                           :d_turbModel(turb_model), 
                            d_physicalConsts(phys_const)
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
  
  db->getWithDefault("whichMMS", d_mms, "linearMMS");
  if (d_mms == "linearMMS") {
    ProblemSpecP db_mms = db->findBlock("linearMMS");
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
  else if (d_mms == "expMMS") {
	  ProblemSpecP db_mms = db->findBlock("expMMS");
	  db_mms->require("cu",cu);
  }
  else if (d_mms == "sineMMS") {
	  ProblemSpecP db_mms = db->findBlock("sineMMS");
	  db_mms->require("cu",cu);
  }
  else
	  throw InvalidValue("current MMS "
		       "not supported: " + d_mms, __FILE__, __LINE__);
}

//****************************************************************************
// Velocity source calculation
//****************************************************************************
void 
Source::calculateVelocitySource(const ProcessorGroup* ,
				const Patch* patch,
				double delta_t,
				int index,
				CellInformation* cellinfo,
				ArchesVariables* vars,
				ArchesConstVariables* constvars)
{
  
  //get index component of gravity
  double gravity = d_physicalConsts->getGravity(index);
  // get iref, jref, kref and ref density by broadcasting from a patch that contains
  // iref, jref and kref

  //  double den_ref = vars->density[IntVector(3,3,3)]; // change it!!! use ipref, jpref and kpref
  //  double den_ref = 1.184344; // change it!!! use ipref, jpref and kpref
  // Get the patch and variable indices
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector idxLoW = patch->getSFCZFORTLowIndex();
  IntVector idxHiW = patch->getSFCZFORTHighIndex();
  // get domain size without ghost cells
  // using ng for no ghost cell
  IntVector domLoUng;
  IntVector domHiUng;
  IntVector domLoVng;
  IntVector domHiVng;
  IntVector domLoWng;
  IntVector domHiWng;

  
  switch(index) {
  case Arches::XDIR:
    domLoUng = vars->uVelLinearSrc.getFortLowIndex();
    domHiUng = vars->uVelLinearSrc.getFortHighIndex();

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


    break;
  case Arches::YDIR:
    domLoVng = vars->vVelLinearSrc.getFortLowIndex();
    domHiVng = vars->vVelLinearSrc.getFortHighIndex();

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


    break;
  case Arches::ZDIR:

    domLoWng = vars->wVelLinearSrc.getFortLowIndex();
    domHiWng = vars->wVelLinearSrc.getFortHighIndex();
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


    break;
  default:
    throw InvalidValue("Invalid index in Source::calcVelSrc", __FILE__, __LINE__);
  }


#ifdef MAY_BE_USEFUL_LATER  
  int ioff = 1;
  int joff = 0;
  int koff = 0;
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  FORT_VELSOURCE(domLoU.get_pointer(), domHiU.get_pointer(),
		 idxLoU.get_pointer(), idxHiU.get_pointer(),
		 uVelLinearSrc.getPointer(), 
		 uVelNonlinearSrc.getPointer(), 
		 uVelocity.getPointer(), 
		 domLoV.get_pointer(), domHiV.get_pointer(),
		 idxLoV.get_pointer(), idxHiV.get_pointer(),
		 vVelLinearSrc.getPointer(), 
		 vVelNonlinearSrc.getPointer(), 
		 vVelocity.getPointer(), 
		 domLoW.get_pointer(), domHiW.get_pointer(),
		 idxLoW.get_pointer(), idxHiW.get_pointer(),
		 wVelLinearSrc.getPointer(), 
		 wVelNonlinearSrc.getPointer(), 
		 wVelocity.getPointer(), 
		 domLo.get_pointer(), domHi.get_pointer(),
		 idxLo.get_pointer(), idxHi.get_pointer(),
		 density.getPointer(),
		 viscosity.getPointer(), 
		 &gravity, 
		 ioff, joff, koff, 
		 cellinfo->ceeu, cellinfo->cweu, cellinfo->cwwu,
		 cellinfo->cnn, cellinfo->csn, cellinfo->css,
		 cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
		 cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		 cellinfo->dxepu, cellinfo->dynp, cellinfo->dztp,
		 cellinfo->dxpw, cellinfo->fac1u, cellinfo->fac2u,
		 cellinfo->fac3u, cellinfo->fac4u,cellinfo->iesdu,
		 cellinfo->iwsdu, cellinfo->enfac, cellinfo->sfac,
		 cellinfo->tfac, cellinfo->bfac, volume);
#endif

  // pass the pointer to turbulence model object and make 
  // it a data memeber of Source class
  // it computes the source in momentum eqn due to the turbulence
  // model used.
  // inputs : 
  // outputs : 
  //  d_turbModel->calcVelocitySource(pc, patch, old_dw, new_dw, index);
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
				    ArchesConstVariables* constvars)
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
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
  for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
	IntVector currcell(ii,jj,kk);
	vars->pressNonlinearSrc[currcell] -= constvars->filterdrhodt[currcell]/delta_t;
      }
    }
  }
  

#if 0
  // correct uvel hat at the boundary
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  if (xplus) {
    int ii = idxHi.x();
    for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {
      for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
	IntVector currcell(ii,jj,kk);
	IntVector nextcell(ii+1,jj,kk);
	double avgden = (vars->density[currcell]+vars->density[nextcell])/0.5;
	double area = cellinfo->sns[jj]*cellinfo->stb[kk]/cellinfo->sew[ii];
	vars->pressNonlinearSrc[currcell] -= 2.0*delta_t*area*
	                                     vars->pressure[currcell];
      }
    }
  }
#endif
#endif
}


//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::calculateScalarSource(const ProcessorGroup*,
			      const Patch* patch,
			      double delta_t,
			      int, 
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
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
				int, 
				CellInformation* cellinfo,
				ArchesVariables* vars,
				ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector indexLow = patch->getCellFORTLowIndex();
  IntVector indexHigh = patch->getCellFORTHighIndex();
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
// Thermal NOx source calculation
// Added by Padmabhushana Desam
//****************************************************************************
void Source::thermalNOxSource(const ProcessorGroup*,
                                const Patch* patch,
                                double,
                                int,
                                CellInformation* cellinfo,
                                ArchesVariables* vars,
                                ArchesConstVariables* constvars)
{
  //double tot_noxsource=0.0; // This varible is for monitoring the total NOx production
  // Get the patch and variable indices
  IntVector indexLow = patch->getCellFORTLowIndex();
  IntVector indexHigh = patch->getCellFORTHighIndex();
  for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ ++) {
    for (int colY = indexLow.y(); colY <= indexHigh.y(); colY ++) {
      for (int colX = indexLow.x(); colX <= indexHigh.x(); colX ++) {
        IntVector currCell(colX, colY, colZ);
        double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
       // There is a multiplication with volume on the left side. So the right hand side als has to be multiplied       // with volume
        vars->scalarNonlinearSrc[currCell] += vol*
                                        constvars->thermalnoxSRC[currCell];
        //tot_noxsource+=vol*constvars->thermalnoxSRC[currCell];
      }
    }
  }
  //cout<<"Total NOx production is:"<<tot_noxsource<<endl;
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
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  fort_scalsrc(idxLo, idxHi, vars->scalarLinearSrc, vars->scalarNonlinearSrc,
	       constvars->old_density, constvars->old_enthalpy,
	       cellinfo->sew, cellinfo->sns, cellinfo->stb, delta_t);

}


//****************************************************************************
// Scalar source calculation
//****************************************************************************
void 
Source::computeEnthalpyRadFluxes(const ProcessorGroup*,
				 const Patch* patch,
				 CellInformation* cellinfo,
				 ArchesVariables* vars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  //  cerr << "temperature before rad flux calculation:" << endl;
  //  vars->temperature.print(cerr);
  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  fort_enthalpyradflux(idxLo, idxHi, vars->qfluxe, vars->qfluxw, vars->qfluxn,
		       vars->qfluxs, vars->qfluxt, vars->qfluxb,
		       vars->temperature, vars->absorption,
		       cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		       cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
#if 0
  cerr << "radiation flux information:" << endl;
  vars->qfluxe.print(cerr);
  cerr << endl << endl;
  vars->qfluxw.print(cerr);
  cerr << endl << endl;
  vars->qfluxn.print(cerr);
  cerr << endl << endl;
  vars->qfluxs.print(cerr);
  cerr << endl << endl;
  vars->qfluxt.print(cerr);
  cerr << endl << endl;
  vars->qfluxb.print(cerr);
  cerr << endl << endl;
#endif
}

void 
Source::computeEnthalpyRadSrc(const ProcessorGroup*,
			      const Patch* patch,
			      CellInformation* cellinfo,
			      ArchesVariables* vars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();

  // 3-d array for volume - fortran uses it for temporary storage
  // Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  fort_enthalpyradsrc(idxLo, idxHi, vars->scalarNonlinearSrc,
		      vars->qfluxe, vars->qfluxw, vars->qfluxn, vars->qfluxs,
		      vars->qfluxt, vars->qfluxb,
		      cellinfo->sew, cellinfo->sns, cellinfo->stb);
#if 0
  cerr << "radiation source after calculation:" << endl;
  vars->scalarNonlinearSrc.print(cerr);
#endif

}

void 
Source::computeEnthalpyRadThinSrc(const ProcessorGroup*,
				  const Patch* patch,
				  CellInformation* cellinfo,
				  ArchesVariables* vars,
				  ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
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
		vars->uVelNonlinearSrc, vars->uVelLinearSrc,
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
		vars->vVelNonlinearSrc, vars->vVelLinearSrc,
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
		vars->wVelNonlinearSrc, vars->wVelLinearSrc,
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
			       int, ArchesVariables* vars,
			       ArchesConstVariables* constvars,
			       int conv_scheme)
{
  // Get the patch and variable indices
  // And call the fortran routine (MASCAL)
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  fort_mascalscalar(idxLo, idxHi, constvars->scalar,
		    vars->scalarCoeff[Arches::AE],
		    vars->scalarCoeff[Arches::AW],
		    vars->scalarCoeff[Arches::AN],
		    vars->scalarCoeff[Arches::AS],
		    vars->scalarCoeff[Arches::AT],
		    vars->scalarCoeff[Arches::AB],
		    vars->scalarNonlinearSrc, vars->scalarLinearSrc,
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
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  fort_mascalscalar(idxLo, idxHi, constvars->enthalpy,
		    vars->scalarCoeff[Arches::AE],
		    vars->scalarCoeff[Arches::AW],
		    vars->scalarCoeff[Arches::AN],
		    vars->scalarCoeff[Arches::AS],
		    vars->scalarCoeff[Arches::AT],
		    vars->scalarCoeff[Arches::AB],
		    vars->scalarNonlinearSrc, vars->scalarLinearSrc,
		    vars->scalarConvectCoeff[Arches::AE],
		    vars->scalarConvectCoeff[Arches::AW],
		    vars->scalarConvectCoeff[Arches::AN],
		    vars->scalarConvectCoeff[Arches::AS],
		    vars->scalarConvectCoeff[Arches::AT],
		    vars->scalarConvectCoeff[Arches::AB],
		    conv_scheme);
}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::addPressureSource(const ProcessorGroup* ,
			  const Patch* patch ,
			  double delta_t,
			  int index,
			  CellInformation* cellinfo,			  
			  ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  int ioff, joff, koff;
  switch(index) {
  case Arches::XDIR:
    domLoU = vars->uVelocity.getFortLowIndex();
    domHiU = vars->uVelocity.getFortHighIndex();
    domLoUng = vars->uVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();

    ioff = 1;
    joff = 0;
    koff = 0;
#ifdef multimaterialform
    Array3<double> bulkVolume(patch->getLowIndex(), patch->getHighIndex());
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();
    MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  // Store current cell
	  IntVector currCell(colX, colY, colZ);
	  bulkVolume[currCell] = cellInfo->sewu[colX]*cellInfo->sns[colY]*
	                         cellInfo->stb[colZ];
	  if (d_mmInterface)
	    bulkVolume[currCell] *= mmVars->voidFraction[currCell];
	}
      }
    }
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->uVelocity.getPointer(),
		      vars->uVelNonlinearSrc.getPointer(), 
		      vars->uVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      bulkVolume.getPointer(),
		      cellinfo->dxpw.get_objs());
#endif    
    fort_addpressgrad(idxLoU, idxHiU, vars->uVelocity, vars->uVelNonlinearSrc,
		      vars->pressure,
		      vars->old_density, delta_t, ioff, joff, koff,
		      cellinfo->sewu, cellinfo->sns, cellinfo->stb,
		      cellinfo->dxpw);
    break;
  case Arches::YDIR:
    domLoU = vars->vVelocity.getFortLowIndex();
    domHiU = vars->vVelocity.getFortHighIndex();
    domLoUng = vars->vVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->vVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0;
    joff = 1;
    koff = 0;
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
#ifdef multimaterialform
    Array3<double> bulkVolume(patch->getLowIndex(), patch->getHighIndex());
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();
    MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  // Store current cell
	  IntVector currCell(colX, colY, colZ);
	  bulkVolume[currCell] = cellInfo->sew[colX]*cellInfo->snsv[colY]*
	                         cellInfo->stb[colZ];
	  if (d_mmInterface)
	    bulkVolume[currCell] *= mmVars->voidFraction[currCell];
	}
      }
    }
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->vVelocity.getPointer(),
		      vars->vVelNonlinearSrc.getPointer(), 
		      vars->vVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      bulkVolume.getPointer(),
		      cellinfo->dyps.get_objs());
#endif    

    fort_addpressgrad(idxLoU, idxHiU, vars->vVelocity, vars->vVelNonlinearSrc,
		      vars->pressure,
		      vars->old_density, delta_t, ioff, joff, koff,
		      cellinfo->sew, cellinfo->snsv, cellinfo->stb,
		      cellinfo->dyps);
    break;
  case Arches::ZDIR:
    domLoU = vars->wVelocity.getFortLowIndex();
    domHiU = vars->wVelocity.getFortHighIndex();
    domLoUng = vars->wVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->wVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    ioff = 0;
    joff = 0;
    koff = 1;
#ifdef multimaterialform
    Array3<double> bulkVolume(patch->getLowIndex(), patch->getHighIndex());
    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();
    MultiMaterialVars* mmVars = d_mmInterface->getMMVars();
    for (int colZ = indexLow.z(); colZ < indexHigh.z(); colZ ++) {
      for (int colY = indexLow.y(); colY < indexHigh.y(); colY ++) {
	for (int colX = indexLow.x(); colX < indexHigh.x(); colX ++) {
	  // Store current cell
	  IntVector currCell(colX, colY, colZ);
	  bulkVolume[currCell] = cellInfo->sew[colX]*cellInfo->sns[colY]*
	                         cellInfo->stbw[colZ];
	  if (d_mmInterface)
	    bulkVolume[currCell] *= mmVars->voidFraction[currCell];
	}
      }
    }
    // set bulkvolume to zero if i,j,k or i-1,j,k equals to multimaterial wall
    FORT_ADDPRESSGRAD(domLoU.get_pointer(), domHiU.get_pointer(),
		      domLoUng.get_pointer(), domHiUng.get_pointer(),
		      idxLoU.get_pointer(), idxHiU.get_pointer(),
		      vars->wVelocity.getPointer(),
		      vars->wVelNonlinearSrc.getPointer(), 
		      vars->wVelocityCoeff[Arches::AP].getPointer(),
		      domLo.get_pointer(), domHi.get_pointer(),
		      domLong.get_pointer(), domHing.get_pointer(),
		      vars->pressure.getPointer(),
		      vars->old_density.getPointer(),
		      &delta_t, &ioff, &joff, &koff,
		      bulkVolume.getPointer();
		      cellinfo->dzpb.get_objs());

#endif    
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_addpressgrad(idxLoU, idxHiU, vars->wVelocity, vars->wVelNonlinearSrc,
		      vars->pressure,
		      vars->old_density, delta_t, ioff, joff, koff,
		      cellinfo->sew, cellinfo->sns, cellinfo->stbw,
		      cellinfo->dzpb);
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcPressGrad", __FILE__, __LINE__);
  }
}


//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::addTransMomSource(const ProcessorGroup* ,
			  const Patch* patch ,
			  double delta_t,
			  int index,
			  CellInformation* cellinfo,			  
			  ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  switch(index) {
  case Arches::XDIR:
    domLoU = vars->uVelocity.getFortLowIndex();
    domHiU = vars->uVelocity.getFortHighIndex();
    domLoUng = vars->uVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->uVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    fort_addtranssrc(idxLoU, idxHiU, vars->uVelocity, vars->uVelNonlinearSrc,
		     vars->uVelocityCoeff[Arches::AP], vars->old_density,
		     delta_t, cellinfo->sewu, cellinfo->sns, cellinfo->stb);
    break;
  case Arches::YDIR:
    domLoU = vars->vVelocity.getFortLowIndex();
    domHiU = vars->vVelocity.getFortHighIndex();
    domLoUng = vars->vVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->vVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    fort_addtranssrc(idxLoU, idxHiU, vars->vVelocity, vars->vVelNonlinearSrc,
		     vars->vVelocityCoeff[Arches::AP], vars->old_density,
		     delta_t, cellinfo->sew, cellinfo->snsv, cellinfo->stb);
    break;
  case Arches::ZDIR:
    domLoU = vars->wVelocity.getFortLowIndex();
    domHiU = vars->wVelocity.getFortHighIndex();
    domLoUng = vars->wVelNonlinearSrc.getFortLowIndex();
    domHiUng = vars->wVelNonlinearSrc.getFortHighIndex();
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    fort_addtranssrc(idxLoU, idxHiU, vars->uVelocity, vars->uVelNonlinearSrc,
		     vars->uVelocityCoeff[Arches::AP], vars->old_density,
		     delta_t, cellinfo->sew, cellinfo->sns, cellinfo->stbw);
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcTrabsSource", __FILE__, __LINE__);
  }
}

//****************************************************************************
// Documentation here
//****************************************************************************
void 
Source::computePressureSource(const ProcessorGroup* ,
			      const Patch* patch ,
			      int index,
			      CellInformation* cellinfo,			  
			      ArchesVariables* vars)
{
  // Get the patch and variable indices
  IntVector domLoU, domHiU;
  IntVector domLoUng, domHiUng;
  IntVector idxLoU, idxHiU;
  int ioff, joff, koff;
  switch(index) {
  case Arches::XDIR:
    domLoU = vars->uVelocity.getFortLowIndex();
    domHiU = vars->uVelocity.getFortHighIndex();
    domLoUng = vars->pressGradUSu.getFortLowIndex();
    domHiUng = vars->pressGradUSu.getFortHighIndex();
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();

    ioff = 1;
    joff = 0;
    koff = 0;
    fort_calcpressgrad(idxLoU, idxHiU, vars->uVelocity, vars->pressGradUSu,
		       vars->pressure, ioff, joff, koff, cellinfo->sewu,
		       cellinfo->sns, cellinfo->stb, cellinfo->dxpw);
    break;
  case Arches::YDIR:
    domLoU = vars->vVelocity.getFortLowIndex();
    domHiU = vars->vVelocity.getFortHighIndex();
    domLoUng = vars->pressGradVSu.getFortLowIndex();
    domHiUng = vars->pressGradVSu.getFortHighIndex();
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0;
    joff = 1;
    koff = 0;
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref

    fort_calcpressgrad(idxLoU, idxHiU, vars->vVelocity, vars->pressGradVSu,
		       vars->pressure, ioff, joff, koff, cellinfo->sew,
		       cellinfo->snsv, cellinfo->stb, cellinfo->dyps);
    break;
  case Arches::ZDIR:
    domLoU = vars->wVelocity.getFortLowIndex();
    domHiU = vars->wVelocity.getFortHighIndex();
    domLoUng = vars->pressGradWSu.getFortLowIndex();
    domHiUng = vars->pressGradWSu.getFortHighIndex();
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
    ioff = 0;
    joff = 0;
    koff = 1;
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_calcpressgrad(idxLoU, idxHiU, vars->wVelocity, vars->pressGradWSu,
		       vars->pressure, ioff, joff, koff, cellinfo->sew,
		       cellinfo->sns, cellinfo->stbw, cellinfo->dzpb);
    break;
  default:
    throw InvalidValue("Invalid index in Source::calcPressGrad", __FILE__, __LINE__);
  }
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
  // for no ghost cells
  IntVector domLoUng;
  IntVector domHiUng;
  IntVector domLo;
  IntVector domHi;
  
  switch(index) {
  case 1:
  // Get the low and high index for the patch and the variables
  idxLoU = patch->getSFCXFORTLowIndex();
  idxHiU = patch->getSFCXFORTHighIndex();
  // for no ghost cells
  domLoUng = vars->uVelLinearSrc.getFortLowIndex();
  domHiUng = vars->uVelLinearSrc.getFortHighIndex();
  domLo = constvars->mmuVelSu.getFortLowIndex();
  domHi = constvars->mmuVelSu.getFortHighIndex();
  

  fort_mmmomsrc(idxLoU, idxHiU, vars->uVelNonlinearSrc, vars->uVelLinearSrc,
		constvars->mmuVelSu, constvars->mmuVelSp);
  break;
  case 2:
  // Get the low and high index for the patch and the variables
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
  // for no ghost cells
    domLoUng = vars->vVelLinearSrc.getFortLowIndex();
    domHiUng = vars->vVelLinearSrc.getFortHighIndex();
    domLo = constvars->mmvVelSu.getFortLowIndex();
    domHi = constvars->mmvVelSu.getFortHighIndex();

    fort_mmmomsrc(idxLoU, idxHiU, vars->vVelNonlinearSrc, vars->vVelLinearSrc,
		  constvars->mmvVelSu, constvars->mmvVelSp);
    break;
  case 3:
  // Get the low and high index for the patch and the variables
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
  // for no ghost cells
    domLoUng = vars->wVelLinearSrc.getFortLowIndex();
    domHiUng = vars->wVelLinearSrc.getFortHighIndex();
    domLo = constvars->mmwVelSu.getFortLowIndex();
    domHi = constvars->mmwVelSu.getFortHighIndex();
  
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

  IntVector valid_lo = patch->getCellFORTLowIndex();
  IntVector valid_hi = patch->getCellFORTHighIndex();

  fort_add_mm_enth_src(vars->scalarNonlinearSrc,
		       vars->scalarLinearSrc,
		       constvars->mmEnthSu,
		       constvars->mmEnthSp,
		       valid_lo,
		       valid_hi);

  //#define enthalpySolve_debug
#ifdef enthalpySolve_debug

    IntVector indexLow = patch->getCellLowIndex();
    IntVector indexHigh = patch->getCellHighIndex();

    int ibot = 0;
    int itop = 0;
    int jbot = 8;
    int jtop = 8;
    int kbot = 8;
    int ktop = 8;

    // values above can be changed for each case as desired

    bool printvalues = true;
    int idloX = Max(indexLow.x(),ibot);
    int idhiX = Min(indexHigh.x()-1,itop);
    int idloY = Max(indexLow.y(),jbot);
    int idhiY = Min(indexHigh.y()-1,jtop);
    int idloZ = Max(indexLow.z(),kbot);
    int idhiZ = Min(indexHigh.z()-1,ktop);
    if ((idloX > idhiX) || (idloY > idhiY) || (idloZ > idhiZ))
      printvalues = false;
    printvalues = false;

    if (printvalues) {
      for (int ii = idloX; ii <= idhiX; ii++) {
	for (int jj = idloY; jj <= idhiY; jj++) {
	  for (int kk = idloZ; kk <= idhiZ; kk++) {
	    cerr.width(14);
	    cerr << " point coordinates "<< ii << " " << jj << " " << kk << endl;
	    cerr << "Before Radiation Solve" << endl;
	    cerr << "Nonlinear source     = " << vars->scalarNonlinearSrc[IntVector(ii,jj,kk)] << endl; 
	  }
	}
      }
    }
#endif

}

//****************************************************************************
// Explicit solve for velocity from projection on hatted velocity
//****************************************************************************

void 
Source::calculateVelocityPred(const ProcessorGroup* ,
			      const Patch* patch,
			      double delta_t,
			      int index,
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars)
{
  
  int ioff, joff, koff;
  IntVector idxLoU;
  IntVector idxHiU;
  IntVector domLoU;
  IntVector domHiU;
  switch(index) {
  case Arches::XDIR:
    idxLoU = patch->getSFCXFORTLowIndex();
    idxHiU = patch->getSFCXFORTHighIndex();
    ioff = 1; joff = 0; koff = 0;
#if 0
    domLoU = vars->uVelRhoHat.getFortLowIndex();
    domHiU = vars->uVelRhoHat.getFortHighIndex();
    cerr << "print before uhat compute" << endl;
    cerr << "print domLoU: " << domLoU << endl;
    cerr << "print domHiU: " << domHiU << endl;
    cerr << "print idxLoU: " << idxLoU << endl;
    cerr << "print idxHiU: " << idxHiU << endl;
    cerr << "print density: " << endl;
    vars->drhopred.print(cerr);
    cerr << "print pressure: " << endl;
    vars->pressure.print(cerr);
    cerr << "print uvelRhoHat: " << endl;
    vars->uVelRhoHat.print(cerr);
#endif
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_computevel(idxLoU, idxHiU, vars->uVelRhoHat, constvars->pressure,
		    constvars->density, delta_t,
		    ioff, joff, koff, cellinfo->dxpw);
#if 0
    cerr << "print uvelRhoHat after solve: " << endl;
    vars->uVelRhoHat.print(cerr);
#endif
    break;
  case Arches::YDIR:
    idxLoU = patch->getSFCYFORTLowIndex();
    idxHiU = patch->getSFCYFORTHighIndex();
    ioff = 0; joff = 1; koff = 0;
#if 0
    domLoU = vars->vVelRhoHat.getFortLowIndex();
    domHiU = vars->vVelRhoHat.getFortHighIndex();
    cerr << "print before vhat compute" << endl;
    cerr << "print domLoU: " << domLoU << endl;
    cerr << "print domHiU: " << domHiU << endl;
    cerr << "print idxLoU: " << idxLoU << endl;
    cerr << "print idxHiU: " << idxHiU << endl;
    cerr << "print density: " << endl;
    vars->drhopred.print(cerr);
    cerr << "print pressure: " << endl;
    vars->pressure.print(cerr);
    cerr << "print vvelRhoHat: " << endl;
    vars->vVelRhoHat.print(cerr);
#endif
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_computevel(idxLoU, idxHiU, vars->vVelRhoHat, constvars->pressure,
		    constvars->density, delta_t,
		    ioff, joff, koff, cellinfo->dyps);

#if 0
    cerr << "print vvelRhoHat after solve: " << endl;
    vars->vVelRhoHat.print(cerr);
#endif
    break;
  case Arches::ZDIR:
    idxLoU = patch->getSFCZFORTLowIndex();
    idxHiU = patch->getSFCZFORTHighIndex();
#if 0 
    domLoU = vars->wVelRhoHat.getFortLowIndex();
    domHiU = vars->wVelRhoHat.getFortHighIndex();
   cerr << "print domLoU: " << domLoU << endl;
    cerr << "print domHiU: " << domHiU << endl;
    cerr << "print idxLoU: " << idxLoU << endl;
    cerr << "print idxHiU: " << idxHiU << endl;
#endif
    ioff = 0; joff = 0; koff = 1;
    // computes remaining diffusion term and also computes 
    // source due to gravity...need to pass ipref, jpref and kpref
    fort_computevel(idxLoU, idxHiU, vars->wVelRhoHat, constvars->pressure,
		    constvars->density, delta_t,
		    ioff, joff, koff, cellinfo->dzpb);

    break;
  default:
    throw InvalidValue("Invalid index in Source::calcVelSrc", __FILE__, __LINE__);
  }

}

//****************************************************************************
// Velocity source calculation for MMS
//****************************************************************************
void 
Source::calculateVelMMSource(const ProcessorGroup* ,
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

  double rho0=d_airDensity*d_heDensity/(phi0*d_airDensity+(1-phi0)*d_heDensity);
  switch(index) {
  case Arches::XDIR:

  for (int colZ = idxLoU.z(); colZ <= idxHiU.z(); colZ ++) {
    for (int colY = idxLoU.y(); colY <= idxHiU.y(); colY ++) {
      for (int colX = idxLoU.x(); colX <= idxHiU.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "linearMMS") {
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
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "linearMMS") {
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
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "linearMMS") {
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
Source::calculateScalarMMSource(const ProcessorGroup*,
			      const Patch* patch,
			      double delta_t,
			      int, 
			      CellInformation* cellinfo,
			      ArchesVariables* vars,
			      ArchesConstVariables* constvars) 
{

  // Get the patch and variable indices
  double rho0=d_airDensity*d_heDensity/(phi0*d_airDensity+(1-phi0)*d_heDensity);
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "linearMMS") {
		vars->scalarNonlinearSrc[currCell] += rho0*phi0*(cu+cv+cw);
	}
      }
    }
  }

}
//****************************************************************************
// Pressure source calculation for MMS
//****************************************************************************
void 
Source::calculatePressMMSourcePred(const ProcessorGroup* ,
				    const Patch* patch,
				    double delta_t,
				    CellInformation* cellinfo,
				    ArchesVariables* vars,
				    ArchesConstVariables* constvars)
{

  // Get the patch and variable indices
  double rho0=d_airDensity*d_heDensity/(phi0*d_airDensity+(1-phi0)*d_heDensity);
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex();
  
  for (int colZ = idxLo.z(); colZ <= idxHi.z(); colZ ++) {
    for (int colY = idxLo.y(); colY <= idxHi.y(); colY ++) {
      for (int colX = idxLo.x(); colX <= idxHi.x(); colX ++) {
	IntVector currCell(colX, colY, colZ);
	double vol = cellinfo->sew[colX]*cellinfo->sns[colY]*cellinfo->stb[colZ];
	if (d_mms == "linearMMS") {
		vars->pressNonlinearSrc[currCell] += rho0*(cu+cv+cw);
	}
      }
    }
  }
}




