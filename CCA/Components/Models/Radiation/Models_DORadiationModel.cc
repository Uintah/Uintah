/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- Models_DORadiationModel.cc --------------------------------------------------

#include <sci_defs/hypre_defs.h>

#include <fstream> // work around compiler bug with RHEL 3

#include <Core/Containers/OffsetArray1.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Time.h>
#include <CCA/Components/Models/Radiation/Models_CellInformationP.h>
#include <CCA/Components/Models/Radiation/Models_RadiationSolver.h>
#include <CCA/Components/Models/Radiation/Models_DORadiationModel.h>
#include <CCA/Components/Models/Radiation/Models_PetscSolver.h>
#ifdef HAVE_HYPRE
#  include <CCA/Components/Models/Radiation/Models_HypreSolver.h>
#endif
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <cmath>

using namespace std;
using namespace Uintah;

#ifndef _WIN32
#  include <CCA/Components/Models/Radiation/fortran/m_rordr_fort.h>
//#  include <CCA/Components/Models/Radiation/fortran/m_rordrss_fort.h>
//#  include <CCA/Components/Models/Radiation/fortran/m_rordrtn_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_radarray_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_radcoef_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_radwsgg_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_radcal_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rdombc_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rdomsolve_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rdomsrc_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rdomflux_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rdombmcalc_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rdomvolq_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rshsolve_fort.h>
#  include <CCA/Components/Models/Radiation/fortran/m_rshresults_fort.h>
#endif
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


void Models_DORadiationModel::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dor_ps = ps->appendChild("DORadiationModel");

  dor_ps->appendElement("ordinates",d_sn);
  dor_ps->appendElement("opl",d_opl);
  dor_ps->appendElement("property_model",d_prop_model);
  dor_ps->appendElement("spherical_harmonics",d_SHRadiationCalc);

  d_linearSolver->outputProblemSpec(dor_ps);
}

//****************************************************************************
// Problem Setup for Models_DORadiationModel
//**************************************************************************** 

void 
Models_DORadiationModel::problemSetup(const ProblemSpecP& params)

{
  ProblemSpecP db = params->findBlock("DORadiationModel");

  test_problems = false;
  int nproblem = 1;

  if (db) {
    db->getWithDefault("ordinates",d_sn,2);
    db->require("opl",d_opl);
    db->getWithDefault("property_model",d_prop_model,"radcoef");
    db->getWithDefault("spherical_harmonics",d_SHRadiationCalc,false);
    db->getWithDefault("test_problem",test_problems,false);
    if (test_problems) 
      db->getWithDefault("test_problem_number",nproblem,1);
  }
  else {
    d_sn=6;
    d_opl=0.18;
  }
  if (d_SHRadiationCalc) {
  cout << "this model (spherical harmonics) does not run in parallel and has been disabled" << endl;
  exit(1);
  }

  lprobone = false;
  lprobtwo = false;
  lprobthree = false;

  if (test_problems) {
    if (nproblem == 1)
      lprobone = true;
    else if (nproblem == 2)
      lprobtwo = true;
    else if (nproblem == 3)
      lprobthree = true;
  }

  if (d_prop_model == "radcoef"){ 
    lradcal = false;
    lwsgg = false;
    lambda = 1;
    lplanckmean = false;
    lpatchmean = false;
  }

  if (d_prop_model == "patchmean"){ 
  cout << "WARNING! Serial and parallel results may deviate for this model" << endl;
    lradcal = true;
    lwsgg = false;
    lambda = 6;
    lplanckmean = false;
    lpatchmean = true;
  }

  if (d_prop_model == "wsggm"){ 
  cout << "this model (wsgg) does not run in parallel and has been disabled" << endl;
  exit(1);
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
  
  d_linearSolver->problemSetup(db, d_SHRadiationCalc);

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

#ifndef _WIN32
  fort_m_rordr(d_sn, oxi, omu, oeta, wt);
#endif
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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  IntVector domLo = patch->getCellLowIndex();
  IntVector domHi = patch->getCellHighIndex();

  // Hottel and Sarofim's empirical soot model.  Soot is
  // only a function of stoichiometry and temperature

  for (int ii = idxLo.x(); ii <= idxHi.x(); ii++) {
    for (int jj = idxLo.y(); jj <= idxHi.y(); jj++) {
      for (int kk = idxLo.z(); kk <= idxHi.z(); kk++) {

        IntVector currCell(ii,jj,kk);
        if (constvars->temperature[currCell] > 1000.0) {
          double bc = vars->mixfrac[currCell]*(84.0/100.0)*constvars->density[currCell];
          double c3 = 0.1;
          double rhosoot = 1950.0;
          double cmw = 12.0;
          double sootFactor = 0.01;

          if (vars->mixfrac[currCell] > 0.1)
            vars->sootVF[currCell] = c3*bc*cmw/rhosoot*sootFactor;
          else
            vars->sootVF[currCell] = 0.0;
        }
      }
    }
  }

#ifndef _WIN32
  int flowField = -1;
  fort_m_radcoef(idxLo, idxHi, 
                 vars->temperature, 
                 vars->co2,
                 vars->h2o,
                 d_opl, 
                 vars->sootVF, // Soot volume fraction
                 vars->ABSKG,  // Absorption coefficent
                 vars->ESRCG,  // 
                 vars->shgamma,
                 cellinfo->xx, 
                 cellinfo->yy, 
                 cellinfo->zz, 
                 fraction, 
                 fractiontwo,
                 lprobone, lprobtwo, lprobthree, 
                 lambda, 
                 lradcal,
                 constvars->cellType,flowField);
#endif
}

//***************************************************************************
// Sets the radiation boundary conditions for the D.O method
//***************************************************************************
void 
Models_DORadiationModel::boundaryCondition(const ProcessorGroup*,
                                           const Patch* patch,
                                           RadiationVariables* vars,
                                           RadiationConstVariables* constvars)
{

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;
    
    // temperature and ABSKG are modified in rdombc
    // cellType is a constVars variable in Arches, but here it is used
    // as ffield throughout the domain (except on boundaries).  
    // When MPMICE is implemented, it may be useful to have a cellType 
    // variable.

    IntVector domLo = patch->getCellLowIndex();
    IntVector domHi = patch->getCellHighIndex();
    IntVector idxLo = patch->getFortranCellLowIndex();
    IntVector idxHi = patch->getFortranCellHighIndex();

    // I am retaining cellType here because we may need to have
    // cellType later for the integrated code, for first-order radiative
    // flux effects    
    // I'm doing the stuff below because we want to set boundary
    // conditions on all boundaries, and the cellTypes for the 
    // boundaries are NOT ffields.  I COULD do this purely by rewriting
    // rdomsolve to use xminus, xplus, etc. to set values for source terms
    // at physical boundaries, but that is a rewrite that I am not willing to
    // do now for fear of introducing new bugs.  In addition, cellType may
    // be a useful variable later.

#ifndef _WIN32
    fort_m_rdombc(idxLo, idxHi, 
                  vars->temperature,
                  vars->ABSKG,
                  constvars->cellType, ffield,
                  xminus, xplus, yminus, yplus, zminus, zplus, 
                  test_problems,
                  lprobone, lprobtwo, lprobthree);
#endif
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
#ifndef _WIN32
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

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
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

  // I am retaining cellType here because we may need to have
  // cellType with MPMICE later, for the first-order radiation
  // effects  (see note above for boundarycondition)

  srcbm.resize(domLo.x(),domHi.x());
  srcbm.initialize(0.0);
  srcpone.resize(domLo.x(),domHi.x());
  srcpone.initialize(0.0);
  qfluxbbm.resize(domLo.x(),domHi.x());
  qfluxbbm.initialize(0.0);

  volume.initialize(0.0);
  volq.initialize(0.0);    
  //  double timeRadMatrix = 0;
  //  double timeRadCoeffs = 0;
  vars->cenint.initialize(0.0);

  //begin discrete ordinates
  if(d_SHRadiationCalc==false) {

    for (int bands =1; bands <=lambda; bands++) {

      volq.initialize(0.0);
      // replacing vars->temperature with constvars->temperature
      if(lwsgg == true){    
        fort_m_radwsgg(idxLo, idxHi, 
                     vars->ABSKG, 
                     vars->ESRCG, 
                     vars->shgamma,
                     bands, 
                     constvars->cellType, ffield, 
                     constvars->co2, 
                     constvars->h2o, 
                     constvars->sootVF, 
                     constvars->temperature, 
                     lambda, 
                     fraction, 
                     fractiontwo);
      }

      // replacing vars->temperature with constvars->temperature
      if(lradcal==true){    
        fort_m_radcal(idxLo, idxHi, 
                    vars->ABSKG, 
                    vars->ESRCG, 
                    vars->shgamma,
                    cellinfo->xx, 
                    cellinfo->yy, 
                    cellinfo->zz, 
                    bands, 
                    constvars->cellType, ffield, 
                    constvars->co2, 
                    constvars->h2o, 
                    constvars->sootVF, 
                    constvars->temperature, 
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

        // rdomsolve sets coefficients and sources for DO linear equation

      // replacing vars->temperature with constvars->temperature
        fort_m_rdomsolve(idxLo, idxHi, 
                         cellinfo->sew,
                         cellinfo->sns, 
                         cellinfo->stb, 
                         vars->ESRCG, direcn, 
                         oxi, 
                         omu,
                         oeta, 
                         wt, 
                         vars->ABSKG, 
                         volume,
                         su,
                         aw, as, ab, 
                         ap, 
                         ae, an, at,
                         constvars->temperature,
                         constvars->cellType, ffield,
                         plusX, plusY, plusZ, 
                         fraction, 
                         bands, 
                         d_opl);

        /*
        if (patch->containsCell(IntVector(11,10,10))) {
          cout << "ap at 11,10,10 = " << ap[IntVector(11,10,10)] << endl;
          cout << "ae at 11,10,10 = " << ae[IntVector(11,10,10)] << endl;
          cout << "aw at 11,10,10 = " << aw[IntVector(11,10,10)] << endl;
          cout << "an at 11,10,10 = " << an[IntVector(11,10,10)] << endl;
          cout << "as at 11,10,10 = " << as[IntVector(11,10,10)] << endl;
          cout << "at at 11,10,10 = " << at[IntVector(11,10,10)] << endl;
          cout << "ab at 11,10,10 = " << ab[IntVector(11,10,10)] << endl;
          cout << "su at 11,10,10 = " << su[IntVector(11,10,10)] << endl;
        }
        */
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

    if (test_problems) {
      fort_m_rdombmcalc(idxLo, idxHi, 
                        constvars->cellType, ffield,
                        cellinfo->xx, 
                        cellinfo->zz, 
                        cellinfo->sew, 
                        cellinfo->sns, 
                        cellinfo->stb, 
                        volume,
                        srcbm, 
                        qfluxbbm, 
                        vars->src, 
                        vars->qfluxe, 
                        vars->qfluxw, 
                        vars->qfluxn, 
                        vars->qfluxs, 
                        vars->qfluxt, 
                        vars->qfluxb, 
                        lprobone, lprobtwo, lprobthree, 
                        srcpone, 
                        volq, 
                        srcsum);
      cerr << "Total radiative source =" << srcsum << " watts\n";
    } 
  }//end discrete ordinates


  if(d_SHRadiationCalc){

    double solve_start = Time::currentSeconds();
    for (int bands =1; bands <=lambda; bands++) {

      // replacing vars->temperature with constvars->temperature

      if(lwsgg == true){    
        fort_m_radwsgg(idxLo, idxHi, 
                     vars->ABSKG, 
                     vars->ESRCG, 
                     vars->shgamma,
                     bands, 
                     constvars->cellType, ffield, 
                     constvars->co2, 
                     constvars->h2o, 
                     constvars->sootVF, 
                     constvars->temperature, 
                     lambda, 
                     fraction, 
                     fractiontwo);
      }

      // replacing vars->temperature with constvars->temperature
      if(lradcal==true){    
        fort_m_radcal(idxLo, idxHi, 
                    vars->ABSKG, 
                    vars->ESRCG, 
                    vars->shgamma,
                    cellinfo->xx, 
                    cellinfo->yy, 
                    cellinfo->zz, 
                    bands, 
                    constvars->cellType, ffield, 
                    constvars->co2, 
                    constvars->h2o, 
                    constvars->sootVF, 
                    constvars->temperature, 
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
                    constvars->cellType, ffield, 
                    cellinfo->sew, 
                    cellinfo->sns, 
                    cellinfo->stb,
                    cellinfo->dxep, 
                    cellinfo->dxpw, 
                    cellinfo->dynp, 
                    cellinfo->dyps, 
                    cellinfo->dztp, 
                    cellinfo->dzpb, 
                    vars->ESRCG, 
                    constvars->temperature, 
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
                      constvars->cellType, ffield,
                      cellinfo->dxep,
                      cellinfo->dxpw,
                      cellinfo->dynp,
                      cellinfo->dyps,
                      cellinfo->dztp,
                      cellinfo->dzpb,
                      constvars->temperature,
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
    if (test_problems) {
      fort_m_rdombmcalc(idxLo, idxHi, 
                        constvars->cellType, ffield,
                        cellinfo->xx, 
                        cellinfo->zz, 
                        cellinfo->sew, 
                        cellinfo->sns, 
                        cellinfo->stb, 
                        volume,
                        srcbm, 
                        qfluxbbm, 
                        vars->src, 
                        vars->qfluxe, 
                        vars->qfluxw, 
                        vars->qfluxn, 
                        vars->qfluxs, 
                        vars->qfluxt, 
                        vars->qfluxb, 
                        lprobone, lprobtwo, lprobthree, 
                        srcpone, 
                        volq, 
                        srcsum);
      cerr << "Total radiative source =" << srcsum << " watts\n";
    } 
  }
#endif
}
