/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- DORadiationModel.cc --------------------------------------------------

#include <sci_defs/hypre_defs.h>

#include <fstream> // work around compiler bug with RHEL 3


#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadPetscSolver.h>
#ifdef HAVE_HYPRE
#include <CCA/Components/Arches/Radiation/RadHypreSolver.h>
#endif
//#include <CCA/Components/Arches/Mixing/Common.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Time.h>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <cmath>
#include <Core/Math/MiscMath.h>


using namespace std;
using namespace Uintah;

#include <CCA/Components/Arches/Radiation/fortran/rordr_fort.h>
//#include <CCA/Components/Arches/Radiation/fortran/rordrss_fort.h>
//#include <CCA/Components/Arches/Radiation/fortran/rordrtn_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radcoef_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radwsgg_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radcal_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdombc_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomflux_fort.h>
//#include <CCA/Components/Arches/Radiation/fortran/rdombmcalc_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomvolq_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rshsolve_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rshresults_fort.h>
//****************************************************************************
// Default constructor for DORadiationModel
//****************************************************************************
DORadiationModel::DORadiationModel(BoundaryCondition* bndry_cond,
                                   const ProcessorGroup* myworld):
                                   RadiationModel(),d_boundaryCondition(bndry_cond),
                                   d_myworld(myworld)
{
  d_linearSolver = 0;
  _props_calculator = 0;
  _using_props_calculator = false; 

}

//****************************************************************************
// Destructor
//****************************************************************************
DORadiationModel::~DORadiationModel()
{
  delete d_linearSolver;
  delete _props_calculator; 
}

//****************************************************************************
// Problem Setup for DORadiationModel
//**************************************************************************** 

void 
DORadiationModel::problemSetup(const ProblemSpecP& params)

{
  ProblemSpecP db = params->findBlock("DORadiationModel");

  string prop_model;

  if ( db->findBlock("property_calculator") ){ 

    std::string calculator_type; 
    db->findBlock("property_calculator")->getAttribute("type",calculator_type);

    if ( calculator_type == "constant" ){ 

      _props_calculator = scinew ConstantProperties(); 

    } else if ( calculator_type == "burns_christon" ){ 

      _props_calculator = scinew BurnsChriston(); 

    } else { 

      throw InvalidValue("Error: Property calculator not recognized.",__FILE__, __LINE__); 

    } 

    ProblemSpecP db_pc = db->findBlock("property_calculator"); 
    _using_props_calculator = _props_calculator->problemSetup( db_pc );

    if ( !_using_props_calculator ) {
      throw InvalidValue("Error: Property calculator specified in input file but I was unable to setup your calculator!",__FILE__, __LINE__); 
    } 
  } 

  if (db) {
    db->getWithDefault("ordinates",d_sn,2);
//    db->getWithDefault("opl",d_xumax,3.0); too sensitive to have default
    db->require("opl",d_opl);
    db->getWithDefault("property_model",prop_model,"radcoef");
    db->getWithDefault("spherical_harmonics",d_SHRadiationCalc,false);
  }
  else {
    d_sn=6;
    d_opl=0.18;
  }
  //  lshradmodel = false;
  if (d_SHRadiationCalc) {
    throw InternalError("Spherical harmonics radiation model does not run in parallel and has been disabled", __FILE__, __LINE__);
  }

  lprobone   = false;
  lprobtwo   = false;
  lprobthree = false;

  if (prop_model == "radcoef"){ 
    lradcal     = false;
    lwsgg       = false;
    lambda      = 1;
    lplanckmean = false;
    lpatchmean  = false;
  }

  if (prop_model == "patchmean"){ 
    cout << "WARNING! Serial and parallel results may deviate for this model" << endl;
    lradcal     = true;
    lwsgg       = false;
    lambda      = 6;
    lplanckmean = false;
    lpatchmean  = true;
  }

  if (prop_model == "wsggm"){ 
    throw InternalError("WSGG radiation model does not run in parallel and has been disabled", __FILE__, __LINE__);
    lradcal       = false;
    lwsgg         = true;
    lambda        = 4;
    lplanckmean   = false;
    lpatchmean    = false;
  }

  fraction.resize(1,100);
  fraction.initialize(0.0);

  fractiontwo.resize(1,100);
  fractiontwo.initialize(0.0);

  computeOrdinatesOPL();

  // ** WARNING ** ffield/Symmetry/sfield/outletfield hardcoded to -1,-3,-4,-5
  // These have been copied from BoundaryCondition.cc
    
  string linear_sol;
  //db->getWithDefault("linear_solver",linear_sol,"petsc");
  db->findBlock("LinearSolver")->getAttribute("type",linear_sol);

  if (linear_sol == "petsc"){ 
    d_linearSolver = scinew RadPetscSolver(d_myworld);
#ifdef HAVE_HYPRE
  }else if (linear_sol == "hypre"){ 
    d_linearSolver = scinew RadHypreSolver(d_myworld);
#endif
  }
  

//  d_linearSolver = scinew RadPetscSolver(d_myworld);
  d_linearSolver->problemSetup(db);

  ffield = -1;
  symtry = -3;
  sfield = -4;
  outletfield = -5;

  db->getWithDefault("wall_temperature", d_wall_temp, 293.0); 
  db->getWithDefault("wall_abskg", d_wall_abskg, 1.0); 
}

void
DORadiationModel::computeOrdinatesOPL() {

  /*
  //  if(lprobone==true){
    d_opl = 1.0*d_opl; 
    //  }
  if(lprobtwo==true){
    d_opl = 1.76;
  }
  */
  
  d_totalOrds = d_sn*(d_sn+2);
// d_totalOrds = 8*d_sn*d_sn;

  omu.resize(1,d_totalOrds + 1);
  oeta.resize(1,d_totalOrds + 1);
  oxi.resize(1,d_totalOrds + 1);
  wt.resize(1,d_totalOrds + 1);
  //  ord.resize(1,(d_sn/2) + 1);

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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  IntVector domLo = patch->getExtraCellLowIndex();
  IntVector domHi = patch->getExtraCellHighIndex();

  /*
      IntVector domLo = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
      IntVector domHi = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

    int startZ = domLo.z();
    if (zminus) startZ++;
    int endZ = domHi.z();
    if (zplus) endZ--;
    int startY = domLo.y();
    if (yminus) startY++;
    int endY = domHi.y();
    if (yplus) endY--;
    int startX = domLo.x();
    if (xminus) startX++;
    int endX = domHi.x();
    if (xplus) endX--;

    IntVector idxLo(startX, startY, startZ);
    IntVector idxHi(endX - 1, endY - 1, endZ -1);
  */

    CCVariable<double> shgamma;
    vars->shgamma.allocate(domLo,domHi);
    vars->shgamma.initialize(0.0);

    fort_radcoef(idxLo, idxHi, vars->temperature, 
                 constvars->co2, constvars->h2o, constvars->cellType, ffield, 
                 d_opl, constvars->sootFV, vars->ABSKP, vars->ABSKG, vars->ESRCG, vars->shgamma,
                 cellinfo->xx, cellinfo->yy, cellinfo->zz, fraction, fractiontwo,
                 lprobone, lprobtwo, lprobthree, lambda, lradcal);

    if (_using_props_calculator){ 

      _props_calculator->computeProps( patch, vars->ABSKG );

    } 


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
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  
  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus)  != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus)  != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus)  != Patch::Neighbor;
    
  fort_rdombc(idxLo, idxHi, constvars->cellType, ffield, vars->temperature,
              vars->ABSKG,
              xminus, xplus, yminus, yplus, zminus, zplus, 
              lprobone, lprobtwo, lprobthree, d_wall_temp, d_wall_abskg );


}
//***************************************************************************
// Solves for intensity in the D.O method
//***************************************************************************
void 
DORadiationModel::intensitysolve(const ProcessorGroup* pg,
                                 const Patch* patch,
                                 CellInformation* cellinfo,
                                 ArchesVariables* vars,
                                 ArchesConstVariables* constvars, 
                                 int wall_type )
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
    fort_radarray(rgamma, sd15, sd, sd7, sd3);
  }

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  IntVector domLo = patch->getExtraCellLowIndex();
  IntVector domHi = patch->getExtraCellHighIndex();
  
  double areaew;

  CCVariable<double> volume;
  CCVariable<double> su;
  CCVariable<double> ae;
  CCVariable<double> aw;
  CCVariable<double> an;
  CCVariable<double> as;
  CCVariable<double> at;
  CCVariable<double> ab;
  CCVariable<double> ap;
  //CCVariable<double> volq;
  CCVariable<double> cenint;
  
  vars->cenint.allocate(domLo,domHi);

  volume.allocate(domLo,domHi);
  su.allocate(domLo,domHi);
  ae.allocate(domLo,domHi);
  aw.allocate(domLo,domHi);
  an.allocate(domLo,domHi);
  as.allocate(domLo,domHi);
  at.allocate(domLo,domHi);
  ab.allocate(domLo,domHi);
  ap.allocate(domLo,domHi);
  //volq.allocate(domLo,domHi);
  
  arean.resize(domLo.x(),domHi.x());
  areatb.resize(domLo.x(),domHi.x());
  
  srcbm.resize(domLo.x(),domHi.x());
  srcbm.initialize(0.0);
  srcpone.resize(domLo.x(),domHi.x());
  srcpone.initialize(0.0);
  qfluxbbm.resize(domLo.x(),domHi.x());
  qfluxbbm.initialize(0.0);

  volume.initialize(0.0);
  vars->volq.initialize(0.0);    
  arean.initialize(0.0);
  areatb.initialize(0.0);
  //  double timeRadMatrix = 0;
  //  double timeRadCoeffs = 0;
  vars->cenint.initialize(0.0);

  //__________________________________
  //begin discrete ordinates
  if(d_SHRadiationCalc==false) {

    for (int bands =1; bands <=lambda; bands++){
      vars->volq.initialize(0.0);

      if(lwsgg == true){    
        fort_radwsgg(idxLo, idxHi, vars->ABSKG, vars->ESRCG, vars->shgamma,
                     bands, constvars->cellType, ffield, 
                     constvars->co2, constvars->h2o, constvars->sootFV, 
                     vars->temperature, lambda, fraction, fractiontwo);
      }

      if(lradcal==true){    
        fort_radcal(idxLo, idxHi, vars->ABSKG, vars->ESRCG, vars->shgamma,
                    cellinfo->xx, cellinfo->yy, cellinfo->zz, bands, 
                    constvars->cellType, ffield, 
                    constvars->co2, constvars->h2o, constvars->sootFV, 
                    vars->temperature, lprobone, lprobtwo, lplanckmean, lpatchmean, lambda, fraction, rgamma, 
                    sd15, sd, sd7, sd3, d_opl);
      }

      for (int direcn = 1; direcn <=d_totalOrds; direcn++){
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
        fort_rdomsolve(idxLo, idxHi, constvars->cellType, wall_type, ffield, cellinfo->sew,
                       cellinfo->sns, cellinfo->stb, vars->ESRCG, direcn, oxi, omu,
                       oeta, wt, 
                       vars->temperature, vars->ABSKG, vars->cenint, volume,
                       su, aw, as, ab, ap, ae, an, at,
                       areaew, arean, areatb, vars->volq, vars->src, 
                       plusX, plusY, plusZ, fraction, fractiontwo, bands, 
                       vars->qfluxe, vars->qfluxw,
                       vars->qfluxn, vars->qfluxs,
                       vars->qfluxt, vars->qfluxb, d_opl);

        //      double timeSetMat = Time::currentSeconds();
        d_linearSolver->setMatrix(pg ,patch, vars, plusX, plusY, 
                                  plusZ, su, ab, as, aw, ap, ae, an, at);
        //      timeRadMatrix += Time::currentSeconds() - timeSetMat;
        bool converged =  d_linearSolver->radLinearSolve();
        if (converged) {
          d_linearSolver->copyRadSoln(patch, vars);
        }else {
          throw InternalError("Radiation solver not converged", __FILE__, __LINE__);
        }
        d_linearSolver->destroyMatrix();

        fort_rdomvolq(idxLo, idxHi, direcn, wt, vars->cenint, vars->volq);
        fort_rdomflux(idxLo, idxHi, direcn, oxi, omu, oeta, wt, vars->cenint,
                      plusX, plusY, plusZ, vars->qfluxe, vars->qfluxw,
                      vars->qfluxn, vars->qfluxs,
                      vars->qfluxt, vars->qfluxb);
      }  // ordinate loop

      fort_rdomsrc(idxLo, idxHi, vars->ABSKG, vars->ESRCG,vars->volq, vars->src);
    }  // bands loop
    
    if(d_myworld->myrank() == 0) {
      cerr << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";
    }
    /*
    fort_rdombmcalc(idxLo, idxHi, constvars->cellType, ffield, cellinfo->xx, cellinfo->zz, cellinfo->sew, cellinfo->sns, cellinfo->stb, volume, areaew, arean, areatb, srcbm, qfluxbbm, vars->src, vars->qfluxe, vars->qfluxw, vars->qfluxn, vars->qfluxs, vars->qfluxt, vars->qfluxb, lprobone, lprobtwo, lprobthree, srcpone, volq, srcsum);

     cerr << "Total radiative source =" << srcsum << " watts\n";
    */
  }  //end discrete ordinates


  //__________________________________
  //
  if(d_SHRadiationCalc){

    double solve_start = Time::currentSeconds();

    for (int bands =1; bands <=lambda; bands++){

      if(lwsgg == true){    
        fort_radwsgg(idxLo, idxHi, vars->ABSKG, vars->ESRCG, vars->shgamma,
                     bands,  constvars->cellType, ffield, constvars->co2, constvars->h2o, constvars->sootFV, 
                     vars->temperature, lambda, fraction, fractiontwo);
      }

      if(lradcal==true){    
        fort_radcal(idxLo, idxHi, vars->ABSKG, vars->ESRCG, vars->shgamma,
                    cellinfo->xx, cellinfo->yy, cellinfo->zz, 
                    bands, constvars->cellType, ffield, 
                    constvars->co2, constvars->h2o, constvars->sootFV, vars->temperature, 
                    lprobone, lprobtwo, lplanckmean, lpatchmean, lambda, fraction, rgamma, sd15, sd, sd7, sd3, d_opl);
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

      fort_rshsolve(idxLo, idxHi, constvars->cellType, wall, ffield, 
                    cellinfo->sew, cellinfo->sns, cellinfo->stb,
                    cellinfo->xx,  cellinfo->yy,  cellinfo->zz,
                    vars->ESRCG, vars->temperature, vars->ABSKG, vars->shgamma, 
                    volume, su, ae, aw, an, as, at, ab, ap,
                    areaew, arean, areatb, vars->volq, vars->src, plusX, plusY, plusZ, fraction, fractiontwo, bands);

      //      double timeSetMat = Time::currentSeconds();
      d_linearSolver->setMatrix(pg ,patch, vars, plusX, plusY, 
                                plusZ, su, ab, as, aw, ap, ae, an, at);
      //      timeRadMatrix += Time::currentSeconds() - timeSetMat;
      bool converged =  d_linearSolver->radLinearSolve();
      if (converged) {
        d_linearSolver->copyRadSoln(patch, vars);
      }else {
        throw InternalError("Radiation solver not converged", __FILE__, __LINE__);
      }
      d_linearSolver->destroyMatrix();

      fort_rshresults(idxLo, idxHi, vars->cenint, vars->volq,
                      constvars->cellType, ffield,
                      cellinfo->xx, cellinfo->yy, cellinfo->zz,
                      vars->temperature,
                      vars->qfluxe, vars->qfluxw,
                      vars->qfluxn, vars->qfluxs,
                      vars->qfluxt, vars->qfluxb,
                      vars->ABSKG, vars->shgamma, vars->ESRCG, vars->src, fraction, fractiontwo, bands);
    }  // bands loop

    if(d_myworld->myrank() == 0) {
      cerr << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";
    }
    /*
     fort_rdombmcalc(idxLo, idxHi, constvars->cellType, ffield, cellinfo->xx, cellinfo->zz, cellinfo->sew, cellinfo->sns, cellinfo->stb, volume, areaew, arean, areatb, srcbm, qfluxbbm, vars->src, vars->qfluxe, vars->qfluxw, vars->qfluxn, vars->qfluxs, vars->qfluxt, vars->qfluxb, lprobone, lprobtwo, lprobthree, srcpone, volq, srcsum);

     cerr << "Total radiative source =" << srcsum << " watts\n";
    */
  } 
}
DORadiationModel::PropertyCalculatorBase::PropertyCalculatorBase(){};
DORadiationModel::PropertyCalculatorBase::~PropertyCalculatorBase(){};
DORadiationModel::ConstantProperties::ConstantProperties(){};
DORadiationModel::ConstantProperties::~ConstantProperties(){};
DORadiationModel::BurnsChriston::BurnsChriston(){};
DORadiationModel::BurnsChriston::~BurnsChriston(){};















