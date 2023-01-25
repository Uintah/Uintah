/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/Models/FluidsBased/MassMomEng_src.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Util/DOUT.hpp>

using namespace Uintah;
using namespace std;

//__________________________________
//  ToDo:
//     - Add bulletproofing for geom object to be within the domain
//______________________________________________________________________
//
MassMomEng_src::MassMomEng_src(const ProcessorGroup* myworld,
                               const MaterialManagerP& materialManager,
                               const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager), d_params(params)
{
  d_matlSet = nullptr;
  Ilb = scinew ICELabel();
  totalMass_srcLabel = nullptr;
  totalEng_srcLabel  =  nullptr;
  d_src = scinew src();
}

//______________________________________________________________________
//
MassMomEng_src::~MassMomEng_src()
{
  delete Ilb;


  if(d_matlSet && d_matlSet->removeReference()){
    delete d_matlSet;
  }


  // regions used
  for(vector<MassMomEng_src::Region*>::iterator iter = d_src->regions.begin();
                                                iter != d_src->regions.end(); iter++){
    MassMomEng_src::Region* region = *iter;
    delete region;
  }

  if( totalMass_srcLabel != 0 ){
    VarLabel::destroy(totalMass_srcLabel);
  }
  if( totalMom_srcLabel != 0 ){
    VarLabel::destroy(totalMom_srcLabel);
  }
  if( totalEng_srcLabel != 0 ){
    VarLabel::destroy(totalEng_srcLabel);
  }
  delete d_src;
}


//______________________________________________________________________
//
MassMomEng_src::Region::Region( GeometryPieceP piece,
                                ProblemSpecP& region_ps)
  : piece(piece)
{

  ProblemSpecP algo_ps = region_ps->findBlock("algorithm");
  std::string algo_str {""};
  algo_ps->getAttribute( "type", algo_str );

  if( algo_str == "fixedPrimitiveValues" ){

    algoType = Region::fixedPrimitiveValues;
    algo_ps->require( "velocity_src",    velocity_src);
    algo_ps->require( "density_src",     density_src);
    algo_ps->require( "temperature_src", temp_src);

    algo_ps->getWithDefault( "timeStart", timeStart, 0.0);
    algo_ps->getWithDefault( "timeStop",  timeStop, 9.e99);
  }

}


//______________________________________________________________________
void MassMomEng_src::problemSetup( GridP&,
                                   const bool isRestart )
{
  ProblemSpecP mme_src_ps = d_params->findBlock("MassMomEng_src");
  d_matl = m_materialManager->parseAndLookupMaterial(mme_src_ps, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matlSet = scinew MaterialSet();
  d_matlSet->addAll(m);
  d_matlSet->addReference();

  totalMass_srcLabel  = VarLabel::create( "TotalMass_src",
                                        sum_vartype::getTypeDescription() );
  totalMom_srcLabel  = VarLabel::create("TotalMom_src",
                                        sumvec_vartype::getTypeDescription() );
  totalEng_srcLabel  = VarLabel::create("TotalEng_src",
                                        sum_vartype::getTypeDescription() );

  //__________________________________
  //  read in the sources
  ProblemSpecP srcs_ps = mme_src_ps->findBlock("sources");

   for ( ProblemSpecP geom_obj_ps = srcs_ps->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

    vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPieceP mainpiece;
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
    }
    else if(pieces.size() > 1){
      mainpiece = scinew UnionGeometryPiece(pieces);
    }
    else {
      mainpiece = pieces[0];
    }

    d_src->regions.push_back( scinew MassMomEng_src::Region( mainpiece, geom_obj_ps ) );
   }

  if( d_src->regions.size() == 0 && !isRestart) {
    throw ProblemSetupException("ERROR MassMomEng_src::problemSetup: a location must be specified", __FILE__, __LINE__);
  }
}

//______________________________________________________________________
void MassMomEng_src::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","mass_momentum_energy_src");
  ProblemSpecP mme_src_ps = model_ps->appendChild("MassMomEng_src");
  mme_src_ps->appendElement( "material",d_matl->getName() );
  //__________________________________
  //  initialization regions
  ProblemSpecP srcs_ps = mme_src_ps->appendChild( "sources" );

  vector<Region*>::const_iterator itr;
  for ( itr = d_src->regions.begin(); itr != d_src->regions.end(); itr++) {
    ProblemSpecP geom_ps = srcs_ps->appendChild( "geom_object" );

    (*itr)->piece->outputProblemSpec(geom_ps);
    ProblemSpecP algo_ps = geom_ps->appendChild( "algorithm" );

    if( (*itr)->algoType == MassMomEng_src::Region::fixedPrimitiveValues ){
      algo_ps->setAttribute( "type", "fixedPrimitiveValues" );

      algo_ps->appendElement( "velocity_src",    (*itr)->velocity_src );
      algo_ps->appendElement( "density_src",     (*itr)->density_src );
      algo_ps->appendElement( "temperature_src", (*itr)->temp_src );
      algo_ps->appendElement( "timeStart",       (*itr)->timeStart );
      algo_ps->appendElement( "timeStop",        (*itr)->timeStop );
    }
  }

}

//______________________________________________________________________
void MassMomEng_src::scheduleInitialize(SchedulerP&,
                                   const LevelP& level)
{
  // None necessary...
}

//______________________________________________________________________
void MassMomEng_src::scheduleComputeStableTimeStep(SchedulerP&,
                                              const LevelP&)
{
  // None necessary...
}

//__________________________________
void MassMomEng_src::scheduleComputeModelSources(SchedulerP& sched,
                                                const LevelP& level)
{
  Task* t = scinew Task("MassMomEng_src::computeModelSources",this,
                        &MassMomEng_src::computeModelSources);
  t->modifies( Ilb->modelMass_srcLabel );
  t->modifies( Ilb->modelMom_srcLabel );
  t->modifies( Ilb->modelEng_srcLabel );
  t->modifies( Ilb->modelVol_srcLabel );

  Ghost::GhostType  gn  = Ghost::None;
  t->requires( Task::OldDW, Ilb->simulationTimeLabel );
  t->requires( Task::OldDW, Ilb->delTLabel,        level.get_rep() );

  t->requires( Task::OldDW, Ilb->rho_CCLabel,        gn );
  t->requires( Task::OldDW, Ilb->temp_CCLabel,       gn );
  t->requires( Task::OldDW, Ilb->vel_CCLabel,        gn );
  t->requires( Task::OldDW, Ilb->specific_heatLabel, gn );

  t->requires( Task::NewDW, Ilb->specific_heatLabel, gn );
  t->requires( Task::NewDW, Ilb->sp_vol_CCLabel,     gn );
  t->requires( Task::NewDW, Ilb->vol_frac_CCLabel,   gn );

  t->computes( MassMomEng_src::totalMass_srcLabel );
  t->computes( MassMomEng_src::totalMom_srcLabel );
  t->computes( MassMomEng_src::totalEng_srcLabel );


  sched->addTask(t, level->eachPatch(), d_matlSet );
}

//__________________________________
void MassMomEng_src::computeModelSources(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, Ilb->simulationTimeLabel);
  double simTime = simTimeVar;

  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel,getLevel(patches));

  int indx = d_matl->getDWIndex();
  double totalMass_src = 0.0;
  double totalEng_src = 0.0;
  Vector totalMom_src(0,0,0);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Vector dx = patch->dCell();
    const double cellVol = dx.x()*dx.y()*dx.z();

    CCVariable<double> mass_src;
    CCVariable<Vector> mom_src;
    CCVariable<double> eng_src;
    CCVariable<double> vol_src;

    constCCVariable<double> sp_vol_CC;
    constCCVariable<double> vol_frac;

    constCCVariable<double> cv;
    constCCVariable<double> cv_old;
    constCCVariable<double> rho_CC_old;
    constCCVariable<double> temp_CC_old;
    constCCVariable<Vector> vel_CC_old;

    new_dw->getModifiable( mass_src, Ilb->modelMass_srcLabel, indx, patch );
    new_dw->getModifiable( mom_src,  Ilb->modelMom_srcLabel,  indx, patch );
    new_dw->getModifiable( eng_src,  Ilb->modelEng_srcLabel,  indx, patch );
    new_dw->getModifiable( vol_src,  Ilb->modelVol_srcLabel,  indx, patch );

    Ghost::GhostType  gn  = Ghost::None;

    old_dw->get( rho_CC_old, Ilb->rho_CCLabel,       indx, patch, gn,0);
    old_dw->get( vel_CC_old, Ilb->vel_CCLabel,       indx, patch, gn,0);
    old_dw->get( temp_CC_old,Ilb->temp_CCLabel,      indx, patch, gn,0);
    old_dw->get( cv_old,     Ilb->specific_heatLabel,indx, patch, gn,0);

    new_dw->get( cv,        Ilb->specific_heatLabel, indx, patch, gn,0);
    new_dw->get( sp_vol_CC, Ilb->sp_vol_CCLabel,     indx, patch, gn,0);
    new_dw->get( vol_frac,  Ilb->vol_frac_CCLabel,   indx, patch, gn,0);

    //__________________________________
    //  Uniform initialization scalar field in a region
    for(auto iter = d_src->regions.begin(); iter != d_src->regions.end(); iter++){
      MassMomEng_src::Region* region = *iter;

      Uintah::Box box  (region->piece->getBoundingBox());
      Uintah::Box patchBox (patch->getBox());

      bool patchContainPiece = ( box.overlaps( patchBox) );

      bool isItTime = (( simTime >= region->timeStart ) && ( simTime <= region->timeStop) );

      bool isAlgorithm = ( region->algoType == MassMomEng_src::Region::fixedPrimitiveValues );

      //__________________________________
      //  Algorithm:  fixed the primitive values
      if( isItTime  &&  patchContainPiece && isAlgorithm){

        //DOUTR( true, "passed conditional  patchID: " << patch->getID() );

        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;

          Point p_lo = patch->nodePosition(c);     // bottom cell corner
          Point p_hi = p_lo + dx;                  // upper cell corner
          
          bool isInside = ( region->piece->inside(p_lo) || 
                            region->piece->inside(p_hi) );

          if ( vol_frac[c] > 0.001 && isInside ) {
          
            //DOUTR( true, "passed conditional  p" << p << " p_hi: " << p_hi );

            const double mass_old = rho_CC_old[c] * cellVol;

            if( region->velocity_src != Vector(0.,0.,0.) ){
              Vector usr_mom_src  = region->velocity_src * mass_old;
              mom_src[c]  += ( usr_mom_src  * vol_frac[c] ) - vel_CC_old[c] * mass_old;
            }

            if( region->temp_src != 0.0 ){
              double usr_eng_src  = region->temp_src * mass_old * cv[c];
              eng_src[c]  += ( usr_eng_src  * vol_frac[c] ) - cv_old[c] * temp_CC_old[c] * mass_old;
            }

            if( region->density_src != 0.0 ){
              double usr_mass_src = region->density_src * cellVol;
              mass_src[c] += ( usr_mass_src * vol_frac[c] ) - mass_old;
  //          vol_src[c]  += mass_src * sp_vol_CC[c]*vol_frac[c];// volume src
            }

            totalMass_src += mass_src[c];
            totalMom_src  += mom_src[c];
            totalEng_src  += eng_src[c];
          }
        } // cellIterator
      } // is it time and patch contains geomObject
    }  // regions
  }  // patches

  new_dw->put( sum_vartype(totalMass_src),    MassMomEng_src::totalMass_srcLabel );
  new_dw->put( sumvec_vartype(totalMom_src),  MassMomEng_src::totalMom_srcLabel );
  new_dw->put( sum_vartype(totalEng_src),     MassMomEng_src::totalEng_srcLabel );
}
//______________________________________________________________________

void MassMomEng_src::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                        const LevelP&,
                                                        const MaterialSet*)
{
  // do nothing
}
void MassMomEng_src::computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,
                                    DataWarehouse*,
                                    const int)
{
  //do nothing
}
//______________________________________________________________________
//
void MassMomEng_src::scheduleErrorEstimate(const LevelP&,
                                      SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void MassMomEng_src::scheduleTestConservation(SchedulerP&,
                                         const PatchSet*)
{
  // Not implemented yet
}
