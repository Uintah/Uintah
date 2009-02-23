/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#include <Packages/Uintah/CCA/Components/Arches/Source.h>
#include <Packages/Uintah/CCA/Components/Arches/Discretization.h>
#include <Packages/Uintah/CCA/Components/Arches/PhysicalConstants.h>
#include <Packages/Uintah/CCA/Components/Arches/BoundaryCondition.h>
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/mascal_scalar_fort.h>
#ifdef divergenceconstraint
#include <Packages/Uintah/CCA/Components/Arches/fortran/pressrcpred_var_fort.h>
#else
#include <Packages/Uintah/CCA/Components/Arches/fortran/pressrcpred_fort.h>
#endif
#include <Packages/Uintah/CCA/Components/Arches/fortran/add_mm_enth_src_fort.h>
#include <Packages/Uintah/CCA/Components/Arches/fortran/enthalpyradthinsrc_fort.h>
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

  if(!db->getAttribute("whichMMS", d_mms))
    d_mms="constantMMS";

  if (d_mms == "constantMMS"){
    ProblemSpecP db_mms = db->findBlock("constantMMS");
    db_mms->getWithDefault("cu",cu,1.0);
    db_mms->getWithDefault("cv",cv,1.0);
    db_mms->getWithDefault("cw",cw,1.0);
    db_mms->getWithDefault("cp",cp,1.0);
    db_mms->getWithDefault("phi0",phi0,0.5);
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
Source::calculateVelocitySource(const Patch* patch,
                                double delta_t,
                                CellInformation* cellinfo,
                                ArchesVariables* vars,
                                ArchesConstVariables* constvars)
{
  
  //get index component of gravity
  Vector gravity = d_physicalConsts->getGravity();
  double grav;
  
  //__________________________________
  //      X DIR  
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  grav = gravity.x();
  
  IntVector shift(-1,0,0);  // ignore outer edge/plane of computational domain
  IntVector idxLoU =  patch->getExtraLowIndex( Patch::XFaceBased,shift);
  IntVector idxHiU =  patch->getExtraHighIndex(Patch::XFaceBased,shift)-IntVector(1,1,1);
  
  fort_uvelsrc(idxLoU, idxHiU, constvars->uVelocity, constvars->old_uVelocity,
               vars->uVelNonlinearSrc, vars->uVelLinearSrc,
               constvars->vVelocity, constvars->wVelocity, constvars->density,
               constvars->viscosity, constvars->old_density,
               constvars->denRefArray,
               grav, delta_t,  cellinfo->ceeu, cellinfo->cweu, 
               cellinfo->cwwu, cellinfo->cnn, cellinfo->csn, cellinfo->css,
               cellinfo->ctt, cellinfo->cbt, cellinfo->cbb, cellinfo->sewu,
               cellinfo->sew, cellinfo->sns, cellinfo->stb, cellinfo->dxpw,
               cellinfo->fac1u, cellinfo->fac2u, cellinfo->fac3u,
               cellinfo->fac4u, cellinfo->iesdu, cellinfo->iwsdu);


  //__________________________________
  //      Y DIR  
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  grav = gravity.y();
  shift = IntVector(0,-1,0);  // ignore outer edge/plane of computational domain
  IntVector idxLoV =  patch->getExtraLowIndex( Patch::YFaceBased,shift);
  IntVector idxHiV =  patch->getExtraHighIndex(Patch::YFaceBased,shift)-IntVector(1,1,1);
  
  fort_vvelsrc(idxLoV, idxHiV, constvars->vVelocity, constvars->old_vVelocity,
               vars->vVelNonlinearSrc, vars->vVelLinearSrc,
               constvars->uVelocity, constvars->wVelocity, constvars->density,
               constvars->viscosity, constvars->old_density,
               constvars->denRefArray,
               grav, delta_t,
               cellinfo->cee, cellinfo->cwe, cellinfo->cww,
               cellinfo->cnnv, cellinfo->csnv, cellinfo->cssv,
               cellinfo->ctt, cellinfo->cbt, cellinfo->cbb,
               cellinfo->sew, cellinfo->snsv, cellinfo->sns, cellinfo->stb,
               cellinfo->dyps, cellinfo->fac1v, cellinfo->fac2v,
               cellinfo->fac3v, cellinfo->fac4v, cellinfo->jnsdv,
               cellinfo->jssdv); 

        
  //__________________________________
  //      Z DIR
  // computes remaining diffusion term and also computes 
  // source due to gravity...need to pass ipref, jpref and kpref
  grav = gravity.z();
  shift = IntVector(0,0,-1);  // ignore outer edge/plane of computational domain
  IntVector idxLoW =  patch->getExtraLowIndex( Patch::ZFaceBased,shift);
  IntVector idxHiW =  patch->getExtraHighIndex(Patch::ZFaceBased,shift)-IntVector(1,1,1);
  
  fort_wvelsrc(idxLoW, idxHiW, constvars->wVelocity, constvars->old_wVelocity,
               vars->wVelNonlinearSrc, vars->wVelLinearSrc,
               constvars->uVelocity, constvars->vVelocity, constvars->density,
               constvars->viscosity, constvars->old_density,
               constvars->denRefArray,
               grav, delta_t,
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
      vars->uVelNonlinearSrc[*iter] += vars->umomBoundarySrc[*iter];
      vars->vVelNonlinearSrc[*iter] += vars->vmomBoundarySrc[*iter];
      vars->wVelNonlinearSrc[*iter] += vars->wmomBoundarySrc[*iter];
    }
  }
  // -- jeremy --
  
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
  
  for(CellIterator iter = patch->getCellIterator__New(); !iter.done(); iter++) {
    IntVector c = *iter;
    vars->pressNonlinearSrc[c] -= constvars->filterdrhodt[c]/delta_t;
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
//-----------------------------------------------
// New scalar source calculation.
// This should replace calculateScalarSource
// **NOTE!**
// This is adding:
// \rho*vol/dt*\phi_t 
// to the RHS.  
void 
Source::calculateScalarSource__new(const ProcessorGroup* pc,
                              const Patch* patch,
                              double delta_t,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars) 
{
  double vol = 0.0;
  double apo = 0.0; 

  for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
    IntVector curr = *iter; 

    vol = cellinfo->sew[curr.x()]*cellinfo->sns[curr.y()]*cellinfo->stb[curr.z()]; 
    apo = constvars->old_density[curr]*vol/delta_t; 
    vars->scalarNonlinearSrc[curr] += apo*constvars->old_scalar[curr];
    
  }

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
  for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();
    double vol = cellinfo->sew[i] * cellinfo->sns[j] * cellinfo->stb[k];
    vars->scalarNonlinearSrc[c] += vol * constvars->reactscalarSRC[c]*
                                         constvars->density[c];
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
// Compute the mass source term due to continuity and utilization of the 
// conservative form of the pde
//****************************************************************************
template<class T> void
Source::compute_massSource(CellIterator iter,
                           const T& vel,
                           StencilMatrix<T>& velCoeff,
                           T& velNonLinearSrc,
                           StencilMatrix<T>& velConvectCoeff) 
{
  //__________________________________
  // examine each element of the matrix
  for(; !iter.done();iter++) { 
    IntVector c = *iter;
    
    double tiny=1e-20;
    
    for(int e = 1; e <= 6; e++){          // N S E W T B
      if( fabs(velCoeff[e][c]) < tiny ){ //  1 2 3 4 5 6
        velConvectCoeff[e][c] = 0.0;
      }
    }
  }
  
  //__________________________________
  // mass src term
  for(iter.reset(); !iter.done();iter++) { 
    IntVector c = *iter;
    double difference = velConvectCoeff[Arches::AN][c] - velConvectCoeff[Arches::AS][c]
                      + velConvectCoeff[Arches::AE][c] - velConvectCoeff[Arches::AW][c]
                      + velConvectCoeff[Arches::AT][c] - velConvectCoeff[Arches::AB][c];
 
    velNonLinearSrc[c] = velNonLinearSrc[c] - difference * vel[c];
    
  }
}

//****************************************************************************
// 
//****************************************************************************
void 
Source::modifyVelMassSource(const Patch* patch,
                            ArchesVariables* vars,
                            ArchesConstVariables* constvars)
{
  //__________________________________
  //    X dir
  CellIterator iter = patch->getSFCXIterator();
  compute_massSource<SFCXVariable<double> >(iter, constvars->uVelocity, 
                                            vars->uVelocityCoeff,
                                            vars->uVelNonlinearSrc, 
                                            vars->uVelocityConvectCoeff);
  //__________________________________
  //    Y dir
  iter = patch->getSFCYIterator();
  compute_massSource<SFCYVariable<double> >(iter, constvars->vVelocity, 
                                            vars->vVelocityCoeff,
                                            vars->vVelNonlinearSrc, 
                                            vars->vVelocityConvectCoeff);  
  //__________________________________
  //    Z dir
  iter = patch->getSFCZIterator();
  compute_massSource<SFCZVariable<double> >(iter, constvars->wVelocity, 
                                            vars->wVelocityCoeff,
                                            vars->wVelNonlinearSrc, 
                                            vars->wVelocityConvectCoeff);
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
//-----------------------------------------
// modify scalar mass source 
// (scalar equivalent to masscal)
//**NOTE** this looks nearly identical to the one used for 
//         velocity except for some reason l2up was turned off
//         for velocity. 
//         Perhaps we can resuse this code for both?
void 
Source::modifyScalarMassSource__new(const ProcessorGroup* ,
                                    const Patch* patch,
                                    double,
                                    ArchesVariables* vars,
                                    ArchesConstVariables* constvars,
                                    int conv_scheme)
{
  for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
    IntVector curr = *iter; 
    
//     div \cdot (\rho u)
    double smp = conv_scheme == 1 ? 0:
                 vars->scalarConvCoef[curr].e - vars->scalarConvCoef[curr].w + 
                 vars->scalarConvCoef[curr].n - vars->scalarConvCoef[curr].s + 
                 vars->scalarConvCoef[curr].t - vars->scalarConvCoef[curr].b;
    
    vars->scalarNonlinearSrc[curr] += -smp*constvars->scalar[curr];
  }
}
//______________________________________________________________________
//
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
                                CellInformation*,
                                ArchesVariables* vars,
                                ArchesConstVariables* constvars)
{
  //__________________________________
  //    X dir
  CellIterator iter = patch->getSFCXIterator__New(); 
  for(; !iter.done();iter++) { 
    IntVector c = *iter;
    vars->uVelNonlinearSrc[c]  += constvars->mmuVelSu[c];
    vars->uVelLinearSrc[c]     += constvars->mmuVelSp[c];
  }
  //__________________________________
  //    Y dir
  iter = patch->getSFCYIterator__New(); 
  for(; !iter.done();iter++) { 
    IntVector c = *iter;
    vars->vVelNonlinearSrc[c]  += constvars->mmvVelSu[c];
    vars->vVelLinearSrc[c]     += constvars->mmvVelSp[c];
  }

  //__________________________________
  //    Z dir
  iter = patch->getSFCZIterator__New(); 
  for(; !iter.done();iter++) { 
    IntVector c = *iter;
    vars->wVelNonlinearSrc[c]  += constvars->mmwVelSu[c];
    vars->wVelLinearSrc[c]     += constvars->mmwVelSp[c];
  }
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
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars)
{
// empty for now
}
//****************************************************************************
// Scalar source calculation for MMS
// Adds an mms source term to RHS. 
//****************************************************************************
void 
Source::calculateScalarMMSSource(const ProcessorGroup*,
                              const Patch* patch,
                              double delta_t,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars) 
{
  double rho0 = 0.0;

  for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
    //waiting for source terms...
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
      }
    }
  }
}

