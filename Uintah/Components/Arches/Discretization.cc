//----- Discretization.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/Discretization.h>
#include <Uintah/Components/Arches/StencilMatrix.h>
#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Grid/Stencil.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/FCVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/SoleVariable.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Grid/Array3.h>
#include <iostream>
using namespace std;

using namespace Uintah::ArchesSpace;
using SCICore::Geometry::Vector;

//****************************************************************************
// Default constructor for Discretization
//****************************************************************************
Discretization::Discretization()
{
  // BB : (tmp) velocity is set as CCVariable (should be FCVariable)
  d_uVelocityLabel = scinew VarLabel("uVelocity",
				    CCVariable<double>::getTypeDescription() );
  d_vVelocityLabel = scinew VarLabel("vVelocity",
				    CCVariable<double>::getTypeDescription() );
  d_wVelocityLabel = scinew VarLabel("wVelocity",
				    CCVariable<double>::getTypeDescription() );
  d_densityLabel = scinew VarLabel("density",
				   CCVariable<double>::getTypeDescription() );
  d_viscosityLabel = scinew VarLabel("viscosity",
				   CCVariable<double>::getTypeDescription() );
  d_scalarLabel = scinew VarLabel("scalar",
				   CCVariable<double>::getTypeDescription() );
  d_pressureLabel = scinew VarLabel("pressure",
				   CCVariable<double>::getTypeDescription() );
  d_uVelCoefLabel = scinew VarLabel("uVelCoef",
				   CCVariable<double>::getTypeDescription() );
  d_vVelCoefLabel = scinew VarLabel("vVelCoef",
				   CCVariable<double>::getTypeDescription() );
  d_wVelCoefLabel = scinew VarLabel("wVelCoef",
				   CCVariable<double>::getTypeDescription() );
  d_uVelConvCoefLabel = scinew VarLabel("uVelConvCoef",
				   CCVariable<double>::getTypeDescription() );
  d_vVelConvCoefLabel = scinew VarLabel("vVelConvCoef",
				   CCVariable<double>::getTypeDescription() );
  d_wVelConvCoefLabel = scinew VarLabel("wVelConvCoef",
				   CCVariable<double>::getTypeDescription() );
  d_presCoefLabel = scinew VarLabel("presCoef",
				   CCVariable<double>::getTypeDescription() );
  d_scalCoefLabel = scinew VarLabel("scalCoef",
				   CCVariable<double>::getTypeDescription() );
  d_uVelLinSrcLabel = scinew VarLabel("uVelLinSrc",
				   CCVariable<double>::getTypeDescription() );
  d_vVelLinSrcLabel = scinew VarLabel("vVelLinSrc",
				   CCVariable<double>::getTypeDescription() );
  d_wVelLinSrcLabel = scinew VarLabel("wVelLinSrc",
				   CCVariable<double>::getTypeDescription() );
  d_presLinSrcLabel = scinew VarLabel("presLinSrc",
				   CCVariable<double>::getTypeDescription() );
  d_scalLinSrcLabel = scinew VarLabel("scalLinSrc",
				   CCVariable<double>::getTypeDescription() );
}

//****************************************************************************
// Destructor
//****************************************************************************
Discretization::~Discretization()
{
}

//****************************************************************************
// Velocity stencil weights
//****************************************************************************
void 
Discretization::calculateVelocityCoeff(const ProcessorContext* pc,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       double delta_t,
				       int index)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  // (** WARNING **) velocity is a FC variable
  CCVariable<double> uVelocity;
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  // (** WARNING **) velocity is a FC variable
  CCVariable<double> vVelocity;
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  // (** WARNING **) velocity is a FC variable
  CCVariable<double> wVelocity;
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;
#endif

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  //7pt stencil declaration
  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > uVelocityCoeff;
  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > vVelocityCoeff;
  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > wVelocityCoeff;
  // convection coeffs
  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > uVelocityConvectCoeff;
  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > vVelocityConvectCoeff;
  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > wVelocityConvectCoeff;

  // Allocate space in new datawarehouse
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->allocate(uVelocityCoeff[ii], d_uVelCoefLabel, ii, patch);
    new_dw->allocate(vVelocityCoeff[ii], d_vVelCoefLabel, ii, patch);
    new_dw->allocate(wVelocityCoeff[ii], d_wVelCoefLabel, ii, patch);
    new_dw->allocate(uVelocityConvectCoeff[ii], d_uVelConvCoefLabel, ii, patch);
    new_dw->allocate(vVelocityConvectCoeff[ii], d_vVelConvCoefLabel, ii, patch);
    new_dw->allocate(wVelocityConvectCoeff[ii], d_wVelConvCoefLabel, ii, patch);
  }

#ifdef WONT_COMPILE_YET

  int ioff = 1;
  int joff = 0;
  int koff = 0;

  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  FORT_VELCOEF(velocity, viscosity, density,
	       uVelocityConvectCoeff, uVelocityCoeff, delta_t,
	       ioff, joff, koff, lowIndex, highIndex,
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

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(uVelocityCoeff[ii], d_uVelCoefLabel, ii, patch);
    new_dw->put(vVelocityCoeff[ii], d_vVelCoefLabel, ii, patch);
    new_dw->put(wVelocityCoeff[ii], d_wVelCoefLabel, ii, patch);
    new_dw->put(uVelocityConvectCoeff[ii], d_uVelConvCoefLabel, ii, patch);
    new_dw->put(vVelocityConvectCoeff[ii], d_vVelConvCoefLabel, ii, patch);
    new_dw->put(wVelocityConvectCoeff[ii], d_wVelConvCoefLabel, ii, patch);
  }
  //new_dw->put(uVelocityCoeff, "VelocityCoeff", patch, index);
  //new_dw->put(uVelocityConvectCoeff, "VelocityConvectCoeff", patch, index);

}


//****************************************************************************
// Pressure stencil weights
//****************************************************************************
void 
Discretization::calculatePressureCoeff(const ProcessorContext*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw,
				       double delta_t)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 0;

  CCVariable<double> pressure;
  old_dw->get(pressure, d_pressureLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // (** WARNING **) velocity is a FC variable
  CCVariable<double> uVelocity;
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  // (** WARNING **) velocity is a FC variable
  CCVariable<double> vVelocity;
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  // (** WARNING **) velocity is a FC variable
  CCVariable<double> wVelocity;
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > uVelCoeff;
  StencilMatrix<CCVariable<double> > vVelCoeff;
  StencilMatrix<CCVariable<double> > wVelCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(uVelCoeff[ii], d_uVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelCoeff[ii], d_vVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelCoeff[ii], d_wVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
  }
  //int index = 1;
  //new_dw->get(uVelCoeff,"VelocityCoeff",patch, 0, index);
  //index++;
  //FCVariable<Vector> vVelCoeff;
  //new_dw->get(vVelCoeff,"VelocityCoeff",patch, 0, index);
  //index++;
  //FCVariable<Vector> wVelCoeff;
  //new_dw->get(wVelCoeff,"VelocityCoeff",patch, 0, index);

#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;
#endif

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // Create vars for new_dw
  StencilMatrix<CCVariable<double> > pressCoeff; //7 point stencil
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->allocate(pressCoeff[ii], d_presCoefLabel, ii, patch);
  }
  //new_dw->allocate(pressCoeff,"pressureCoeff",patch, 0);

#ifdef WONT_COMPILE_YET
  FORT_PRESSSOURCE(pressCoeff, pressure, velocity, density
		   uVelocityCoeff, vVelocityCoeff, wVelocityCoeff,
		   lowIndex, highIndex,
		   cellinfo->sew, cellinfo->sns, cellinfo->stb,
		   cellinfo->dxep, cellinfo->dxpw, cellinfo->dynp,
		   cellinfo->dyps, cellinfo->dztp, cellinfo->dzpb);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressCoeff[ii], d_presCoefLabel, ii, patch);
  }
  //new_dw->put(pressCoeff, "pressureCoeff", patch, 0);
}
  
//****************************************************************************
// Scalar stencil weights
//****************************************************************************
void 
Discretization::calculateScalarCoeff(const ProcessorContext* pc,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     double delta_t,
				     int index)
{
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  // (** WARNING **) velocity is a FC variable
  CCVariable<double> uVelocity;
  old_dw->get(uVelocity, d_uVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> vVelocity;
  old_dw->get(vVelocity, d_vVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  CCVariable<double> wVelocity;
  old_dw->get(wVelocity, d_wVelocityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> density;
  old_dw->get(density, d_densityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  CCVariable<double> viscosity;
  old_dw->get(viscosity, d_viscosityLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  // ithe componenet of scalar vector
  CCVariable<double> scalar;
  old_dw->get(scalar, d_scalarLabel, index, patch, Ghost::None,
	      numGhostCells);

#ifdef WONT_COMPILE_YET
  // using chain of responsibility pattern for getting cell information
  DataWarehouseP top_dw = new_dw->getTop();
  PerPatch<CellInformation*> cellinfop;
  if(top_dw->exists("cellinfo", patch)){
    top_dw->get(cellinfop, "cellinfo", patch);
  } else {
    cellinfop.setData(scinew CellInformation(patch));
    top_dw->put(cellinfop, "cellinfo", patch);
  } 
  CellInformation* cellinfo = cellinfop;
#endif

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  //7pt stencil declaration
  StencilMatrix<CCVariable<double> > scalarCoeff;

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->allocate(scalarCoeff[ii], d_scalCoefLabel, ii, patch);
  }

#ifdef WONT_COMPILE_YET
  // 3-d array for volume - fortran uses it for temporary storage
  Array3<double> volume(patch->getLowIndex(), patch->getHighIndex());
  FORT_SCALARCOEF(scalarCoeff, scalar, velocity, viscosity, density,
		  delta_t, lowIndex, highIndex,
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

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(scalarCoeff[ii], d_scalCoefLabel, ii, patch);
  }
  //new_dw->put(scalarCoeff, "ScalarCoeff", patch, index);
}

//****************************************************************************
// Calculate the diagonal terms (velocity)
//****************************************************************************
void 
Discretization::calculateVelDiagonal(const ProcessorContext*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw,
				     int index)
{
  
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  // (** WARNING **) velocity is a FC variable
  StencilMatrix<CCVariable<double> > uVelCoeff;
  StencilMatrix<CCVariable<double> > vVelCoeff;
  StencilMatrix<CCVariable<double> > wVelCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(uVelCoeff[ii], d_uVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
    new_dw->get(vVelCoeff[ii], d_vVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
    new_dw->get(wVelCoeff[ii], d_wVelCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
  }
  //CCVariable<double> uVelCoeff;
  //new_dw->get(uVelCoeff, "VelocityCoeff", patch, index, 0);

  // (** WARNING **) velocity is a FC variable
  CCVariable<double> uVelLinearSrc;
  CCVariable<double> vVelLinearSrc;
  CCVariable<double> wVelLinearSrc;
  new_dw->get(uVelLinearSrc, d_uVelLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(vVelLinearSrc, d_vVelLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  new_dw->get(wVelLinearSrc, d_wVelLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);

  //FCVariable<double> uVelLinearSrc;
  //new_dw->get(uVelLinearSrc, "VelLinearSrc", patch, index, 0);
#ifdef WONT_COMPILE_YET
  FORT_APCAL(uVelCoeffvelocity, uVelLinearSrc, lowIndex, highIndex);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(uVelCoeff[ii], d_uVelCoefLabel, ii, patch);
    new_dw->put(vVelCoeff[ii], d_vVelCoefLabel, ii, patch);
    new_dw->put(wVelCoeff[ii], d_wVelCoefLabel, ii, patch);
  }
  //new_dw->put(uVelCoeff, "VelocityCoeff", patch, index, 0);

}

//****************************************************************************
// Pressure diagonal
//****************************************************************************
void 
Discretization::calculatePressDiagonal(const ProcessorContext*,
				       const Patch* patch,
				       DataWarehouseP& old_dw,
				       DataWarehouseP& new_dw) 
{
  
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  StencilMatrix<CCVariable<double> > pressCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(pressCoeff[ii], d_presCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
  }
  //Stencil<double> pressCoeff;
  //new_dw->get(pressCoeff, "PressureCoCoeff", patch, 0);

  CCVariable<double> presLinearSrc;
  new_dw->get(presLinearSrc, d_presLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<double> pressLinearSrc;
  //new_dw->get(pressLinearSrc, "pressureLinearSource", patch, 0);

#ifdef WONT_COMPILE_YET
  FORT_APCAL(pressCoeff, pressLinearSrc, lowIndex, highIndex);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(pressCoeff[ii], d_presCoefLabel, ii, patch);
  }
  //new_dw->put(pressCoeff, "pressureLinearSource", patch, 0);
}

//****************************************************************************
// Scalar diagonal
//****************************************************************************
void 
Discretization::calculateScalarDiagonal(const ProcessorContext*,
					const Patch* patch,
					DataWarehouseP& old_dw,
					DataWarehouseP& new_dw,
					int index)
{
  
  int matlIndex = 0;
  int numGhostCells = 0;
  int nofStencils = 7;

  IntVector lowIndex = patch->getCellLowIndex();
  IntVector highIndex = patch->getCellHighIndex();

  StencilMatrix<CCVariable<double> > scalarCoeff;
  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->get(scalarCoeff[ii], d_scalCoefLabel, ii, patch, Ghost::None,
		numGhostCells);
  }
  //Stencil<double> scalarCoeff;
  //new_dw->get(scalarCoeff, "ScalarCoeff", patch, index, 0);

  CCVariable<double> scalarLinearSrc;
  new_dw->get(scalarLinearSrc, d_scalLinSrcLabel, matlIndex, patch, Ghost::None,
	      numGhostCells);
  //FCVariable<double> scalarLinearSrc;
  //new_dw->get(scalarLinearSrc, "ScalarLinearSource", patch, index, 0);

#ifdef WONT_COMPILE_YET
  FORT_APCAL(scalarCoeff, scalarLinearSrc, lowIndex, highIndex);
#endif

  for (int ii = 0; ii < nofStencils; ii++) {
    new_dw->put(scalarCoeff[ii], d_scalCoefLabel, ii, patch);
  }
  //new_dw->put(scalarCoeff, "ScalarCoeff", patch, index, 0);
}

//
// $Log$
// Revision 1.14  2000/06/14 20:40:48  rawat
// modified boundarycondition for physical boundaries and
// added CellInformation class
//
// Revision 1.13  2000/06/13 06:02:31  bbanerje
// Added some more StencilMatrices and vector<CCVariable> types.
//
// Revision 1.12  2000/06/07 06:13:54  bbanerje
// Changed CCVariable<Vector> to CCVariable<double> for most cases.
// Some of these variables may not be 3D Vectors .. they may be Stencils
// or more than 3D arrays. Need help here.
//
// Revision 1.11  2000/06/04 22:40:13  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
