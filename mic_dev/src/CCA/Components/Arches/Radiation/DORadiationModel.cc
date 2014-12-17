/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

//----- DORadiationModel.cc --------------------------------------------------
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadPetscSolver.h>
#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Thread/Time.h>
#include <cmath>
#include <sci_defs/hypre_defs.h>
#include <iomanip>

#ifdef HAVE_HYPRE
#  include <CCA/Components/Arches/Radiation/RadHypreSolver.h>
#endif

#include <CCA/Components/Arches/Radiation/fortran/rordr_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radcoef_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radwsgg_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radcal_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsrcscattering_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomflux_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomvolq_fort.h>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES_RADIATION",false);

//****************************************************************************
// Default constructor for DORadiationModel
//****************************************************************************
DORadiationModel::DORadiationModel(const ArchesLabel* label,
                                   const MPMArchesLabel* MAlab,
                                   BoundaryCondition* bndry_cond,
                                   const ProcessorGroup* myworld):
                                   d_lab(label),
                                   d_MAlab(MAlab), 
                                   d_boundaryCondition(bndry_cond),
                                   d_myworld(myworld) 
{

  d_linearSolver = 0;
  d_perproc_patches = 0;

}

//****************************************************************************
// Destructor
//****************************************************************************
DORadiationModel::~DORadiationModel()
{

  delete d_linearSolver;

  if(d_perproc_patches && d_perproc_patches->removeReference()){
    delete d_perproc_patches; 
  }

}

//****************************************************************************
// Problem Setup for DORadiationModel
//**************************************************************************** 

void 
DORadiationModel::problemSetup( ProblemSpecP& params )

{

  ProblemSpecP db = params->findBlock("DORadiationModel");

  db->getWithDefault("ReflectOn",reflectionsTurnedOn,false);  //  reflections are off by default.

  //db->getRootNode()->findBlock("Grid")->findBlock("BoundaryConditions") 

  db->getWithDefault("usePrevInt",usePreviousIntensity,false); //  using the previous solve as initial guess, is off by default

  db->getWithDefault("ScatteringOn",_scatteringOn,false); //  using the previous solve as initial guess, is off by default

  if (db) {
    bool ordinates_specified =db->findBlock("ordinates");
    db->getWithDefault("ordinates",d_sn,2);
    if (ordinates_specified == false){
      proc0cout << " Notice: No ordinate number specified.  Defaulting to 2." << endl;
    }
  } else {
    throw ProblemSetupException("Error: <DORadiation> node not found.", __FILE__, __LINE__);
  }

  //WARNING: Hack -- Hard-coded for now. 
  d_lambda      = 1;

  fraction.resize(1,100);
  fraction.initialize(0.0);
  fraction[1]=1.0;  // This a hack to fix DORad with the new property model interface - Derek 6-14

  computeOrdinatesOPL();

  d_print_all_info = false; 
  if ( db->findBlock("print_all_info") ){
    d_print_all_info = true; 
  }
    
  string linear_sol;
  db->findBlock("LinearSolver")->getAttribute("type",linear_sol);

  if (linear_sol == "petsc"){ 

    d_linearSolver = scinew RadPetscSolver(d_myworld);

  } else if (linear_sol == "hypre"){ 

    d_linearSolver = scinew RadHypreSolver(d_myworld);

  }
  
  d_linearSolver->problemSetup(db);
 
  //WARNING: Hack -- flow cells set to -1
  ffield = -1;

  //NOTE: Setting wall properties to 1.0 
  d_intrusion_abskg = 1.0;

  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  for( int ix=0;  ix<d_totalOrds ;ix++){
    ostringstream my_stringstream_object;
    my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix ;
    _IntensityLabels.push_back(  VarLabel::create(my_stringstream_object.str(),  CC_double));
    if(needIntensitiesBool()== false){
     break;  // gets labels for all intensities, otherwise only create 1 label
    }
  }

  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxE"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxW"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxN"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxS"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxT"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxB"));


}
//______________________________________________________________________
//
void
DORadiationModel::computeOrdinatesOPL() {

  d_totalOrds = d_sn*(d_sn+2);

  omu.resize( 1,d_totalOrds + 1);
  oeta.resize(1,d_totalOrds + 1);
  oxi.resize( 1,d_totalOrds + 1);
  wt.resize(  1,d_totalOrds + 1);

  omu.initialize(0.0);
  oeta.initialize(0.0);
  oxi.initialize(0.0);
  wt.initialize(0.0);

  fort_rordr(d_sn, oxi, omu, oeta, wt);

  if (_scatteringOn){
    cosineTheta    = vector<vector< double > > (d_totalOrds,vector<double>(d_totalOrds,0.0));
    solidAngleQuad = vector<vector< double > > (d_totalOrds,vector<double>(d_totalOrds,0.0));

    for (int i=0; i<d_totalOrds ; i++){
      for (int j=0; j<d_totalOrds ; j++){
        cosineTheta[i][j]=oxi[j+1]*oxi[i+1]+oeta[j+1]*oeta[i+1]+omu[j+1]*omu[i+1];
        solidAngleQuad[i][j]=  wt[i+1]/(4.0 * M_PI);
      }
    }
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
 
  //This should be done in the property calculator
//  //__________________________________
//  // loop over computational domain faces
//  vector<Patch::FaceType> bf;
//  patch->getBoundaryFaces(bf);
//  
//  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
//    Patch::FaceType face = *iter;
//    
//    Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
//    
//    for (CellIterator iter =  patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
//      IntVector c = *iter;
//      if (constvars->cellType[c] != ffield ){
//        vars->ABSKG[c]       = d_wall_abskg;
//      }
//    }
//  }

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
                                 CCVariable<double>& divQ,
                                 int wall_type, int matlIndex,  
                                 DataWarehouse* new_dw, 
                                 DataWarehouse* old_dw)
{

  proc0cout << " Radiation Solve: " << endl;

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

  if (d_lambda > 1) {
    fort_radarray(rgamma, sd15, sd, sd7, sd3);
  }
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  IntVector domLo = patch->getExtraCellLowIndex();
  IntVector domHi = patch->getExtraCellHighIndex();

  CCVariable<double> su;
  CCVariable<double> ae;
  CCVariable<double> aw;
  CCVariable<double> an;
  CCVariable<double> as;
  CCVariable<double> at;
  CCVariable<double> ab;
  CCVariable<double> ap;

  StaticArray< CCVariable<double> > radiationFlux_old(_radiationFluxLabels.size()); // must always 6, even when reflections are off.

  if(reflectionsTurnedOn){
    for (unsigned int i=0; i<  _radiationFluxLabels.size(); i++){
      constCCVariable<double>  radiationFlux_temp;
      old_dw->get(radiationFlux_temp,_radiationFluxLabels[i], matlIndex , patch,Ghost::None, 0  );
      radiationFlux_old[i].allocate(domLo,domHi);
      radiationFlux_old[i].copyData(radiationFlux_temp);
    }
  }
  else{
    for (unsigned int i=0; i<  _radiationFluxLabels.size(); i++){  // magic number cooresponds to number of labels tranported, when 
      radiationFlux_old[i].allocate(domLo,domHi);
      radiationFlux_old[i].initialize(0.0);      // for no reflections, this must be zero
    }
  }
  

  if(usePreviousIntensity==false){
    old_dw->get(constvars->cenint,_IntensityLabels[0], matlIndex , patch,Ghost::None, 0  );
    new_dw->getModifiable(vars->cenint,_IntensityLabels[0] , matlIndex, patch ); // per the logic in sourceterms/doradiation, old and new dw are the same.
  }


  StaticArray< constCCVariable<double> > Intensities(_scatteringOn ? d_totalOrds : 0);

  CCVariable<double> scatIntensitySource;  
  constCCVariable<double> scatkt;   //total scattering coefficient
  constCCVariable<double> asymmetryParam;   //total scattering coefficient

  scatIntensitySource.allocate(domLo,domHi);
  scatIntensitySource.initialize(0.0); // needed for non-scattering cases


  if(_scatteringOn){
    for( int ix=0;  ix<d_totalOrds ;ix++)
      old_dw->get(Intensities[ix],_IntensityLabels[ix], matlIndex , patch,Ghost::None, 0  );
      old_dw->get(asymmetryParam,_asymmetryLabel, matlIndex , patch,Ghost::None, 0);
      old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);
    }


  su.allocate(domLo,domHi);
  ae.allocate(domLo,domHi);
  aw.allocate(domLo,domHi);
  an.allocate(domLo,domHi);
  as.allocate(domLo,domHi);
  at.allocate(domLo,domHi);
  ab.allocate(domLo,domHi);
  ap.allocate(domLo,domHi);
  
  srcbm.resize(domLo.x(),domHi.x());
  srcbm.initialize(0.0);
  srcpone.resize(domLo.x(),domHi.x());
  srcpone.initialize(0.0);
  qfluxbbm.resize(domLo.x(),domHi.x());
  qfluxbbm.initialize(0.0);

  divQ.initialize(0.0);
  vars->qfluxe.initialize(0.0);
  vars->qfluxw.initialize(0.0);
  vars->qfluxn.initialize(0.0);
  vars->qfluxs.initialize(0.0);
  vars->qfluxt.initialize(0.0);
  vars->qfluxb.initialize(0.0);

  //__________________________________
  //begin discrete ordinates
  for (int bands =1; bands <=d_lambda; bands++){

    vars->volq.initialize(0.0);

    for (int direcn = 1; direcn <=d_totalOrds; direcn++){
      if(usePreviousIntensity ){
        old_dw->get(constvars->cenint,_IntensityLabels[direcn-1], matlIndex , patch,Ghost::None, 0  );
        new_dw->getModifiable(vars->cenint,_IntensityLabels[direcn-1] , matlIndex, patch );
      }
      else if ( _scatteringOn){
        new_dw->getModifiable(vars->cenint,_IntensityLabels[direcn-1] , matlIndex, patch );
       }
  
    vars->cenint.initialize(0.0); // remove once RTs have been checked.


      
      su.initialize(0.0);
      aw.initialize(0.0);
      as.initialize(0.0);
      ab.initialize(0.0);
      ap.initialize(0.0);
      ae.initialize(0.0);
      an.initialize(0.0);
      at.initialize(0.0);
      bool plusX, plusY, plusZ;

      if(_scatteringOn){
        scatIntensitySource.initialize(0.0); // needed for summation
        computeScatteringIntensities(direcn, constvars->ABSKG, Intensities,scatIntensitySource,asymmetryParam, patch);
      }
                                                         

      fort_rdomsolve( idxLo, idxHi, constvars->cellType, ffield, 
                      cellinfo->sew, cellinfo->sns, cellinfo->stb, 
                      vars->ESRCG, direcn, oxi, omu,oeta, wt, 
                      constvars->temperature, constvars->ABSKG,
                      su, aw, as, ab, ap, ae, an, at,
                      plusX, plusY, plusZ, fraction, bands, 
                      d_intrusion_abskg,
                      radiationFlux_old[0] , radiationFlux_old[1],
                      radiationFlux_old[2] , radiationFlux_old[3],
                      radiationFlux_old[4] , radiationFlux_old[5],
                      scatIntensitySource); //  this term needed for scattering



      d_linearSolver->setMatrix( pg ,patch, vars, constvars, plusX, plusY, plusZ, 
                                 su, ab, as, aw, ap, ae, an, at, d_print_all_info );
                                
      bool converged =  d_linearSolver->radLinearSolve( direcn, d_print_all_info );
      
      if (converged) {
        d_linearSolver->copyRadSoln(patch, vars);
      }else {
        throw InternalError("Radiation solver not converged", __FILE__, __LINE__);
      }
      
      d_linearSolver->destroyMatrix();

      fort_rdomvolq( idxLo, idxHi, direcn, wt, vars->cenint, vars->volq);

      fort_rdomflux( idxLo, idxHi, direcn, oxi, omu, oeta, wt, vars->cenint,
                     plusX, plusY, plusZ, 
                     vars->qfluxe, vars->qfluxw,
                     vars->qfluxn, vars->qfluxs,
                     vars->qfluxt, vars->qfluxb);
                     
    }  // ordinate loop
 
  if(_scatteringOn)
    fort_rdomsrcscattering( idxLo, idxHi, constvars->ABSKG, vars->ESRCG,vars->volq, divQ, scatkt); 
  else
    fort_rdomsrc( idxLo, idxHi, constvars->ABSKG, vars->ESRCG,vars->volq, divQ); 


  }  // bands loop

  proc0cout << "Total Radiation Solve Time: " << Time::currentSeconds()-solve_start << " seconds\n";

}
// returns the total number of directions, sn*(sn+2)
int 
DORadiationModel::getIntOrdinates(){
return d_totalOrds;
}

// Do the walls reflect? (should only be off if emissivity of walls = 1.0)
bool 
DORadiationModel::reflectionsBool(){
return reflectionsTurnedOn;
}

// Do the Intensities need to be saved from the previous solve?
// Yes, if we are using the previous intensity as our initial guess in the linear solve.
// Yes, if we modeling scattering physics, by lagging the scattering source term.
bool 
DORadiationModel::needIntensitiesBool(){
return usePreviousIntensity || _scatteringOn  ;
}

// Model scattering physics of particles?
bool 
DORadiationModel::ScatteringOnBool(){
return _scatteringOn;
}

void
DORadiationModel::setLabels(){
   _scatktLabel= VarLabel::find("scatkt");
   _asymmetryLabel=VarLabel::find("asymmetryParam"); 
return ;
}

void
DORadiationModel::computeScatteringIntensities(int direction, constCCVariable<double> &scatkt, StaticArray < constCCVariable<double> > &Intensities, CCVariable<double> &scatIntensitySource,constCCVariable<double> &asymmetryFactor , const Patch* patch ){
  direction -=1;   // change from fortran vector to c++ vector
  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

    if (scatkt[*iter] < 1e-6) // intended to increase speed! 
      continue;

    for (int i=0; i < d_totalOrds ; i++) {                                    
      double phaseFunction = (1.0 + asymmetryFactor[*iter]*cosineTheta[direction][i])* solidAngleQuad[direction][i];      
      scatIntensitySource[*iter]  +=phaseFunction*Intensities[i][*iter] ; // wt could be comuted up with the phase function in the j loop

    }

  }

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    scatIntensitySource[*iter] *= scatkt[*iter]  ;
  }

  return;
}
