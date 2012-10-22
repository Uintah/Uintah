/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


#include <CCA/Components/Models/FluidsBased/Mixing.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>
using namespace Uintah;
using namespace std;

Mixing::Mixing(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
}

Mixing::~Mixing()
{
  if(mymatls && mymatls->removeReference())
    delete mymatls;
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    VarLabel::destroy(stream->massFraction_CCLabel);
    VarLabel::destroy(stream->massFraction_source_CCLabel);
    for(vector<Region*>::iterator iter = stream->regions.begin();
        iter != stream->regions.end(); iter++){
      Region* region = *iter;
      delete region;
    }
  }
}

Mixing::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("massFraction", initialMassFraction);
}

void Mixing::outputProblemSpec(ProblemSpecP& ps)
{

  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","Mixing");

  model_ps->appendElement("material",matl->getName());


  for (vector<Stream*>::const_iterator s_it = streams.begin(); 
       s_it != streams.end(); s_it++) {
    ProblemSpecP stream_ps = model_ps->appendChild("stream");
    stream_ps->setAttribute("name",(*s_it)->name);
    (*s_it)->props.outputProblemSpec(stream_ps);

    for (vector<Region*>::const_iterator r_it = (*s_it)->regions.begin();
         r_it != (*s_it)->regions.end(); r_it++) {
      ProblemSpecP geom_ps = stream_ps->appendChild("geom_object");
      (*r_it)->piece->outputProblemSpec(geom_ps);
    }
  }

  for (vector<Reaction*>::const_iterator r_it = reactions.begin(); 
       r_it != reactions.end(); r_it++) {
    ProblemSpecP reaction_ps = model_ps->appendChild("reaction");
    reaction_ps->appendElement("from",(*r_it)->fromStream);
    reaction_ps->appendElement("to",(*r_it)->toStream);

  }

}

void Mixing::problemSetup(GridP&, SimulationStateP& sharedState,
                          ModelSetup* setup)
{
  matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = matl->getDWIndex();
  mymatls = scinew MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();

  // Parse the streams
  int index = 0;
  for (ProblemSpecP child = params->findBlock("stream"); child != 0;
       child = child->findNextBlock("stream")) {
    Stream* stream = scinew Stream();
    stream->index = index++;
    child->getAttribute("name", stream->name);
    string mfname = "massFraction-"+stream->name;
    stream->massFraction_CCLabel = VarLabel::create(mfname, CCVariable<double>::getTypeDescription());
    string mfsname = "massFractionSource-"+stream->name;
    stream->massFraction_source_CCLabel = VarLabel::create(mfsname, CCVariable<double>::getTypeDescription());
    stream->props.parse(child);

    int count = 0;
    for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
         geom_obj_ps != 0;
         geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
      
      vector<GeometryPieceP> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);
      
      GeometryPieceP mainpiece;
      if(pieces.size() == 0){
        throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
      } else if(pieces.size() > 1){
        mainpiece = scinew UnionGeometryPiece(pieces);
      } else {
        mainpiece = pieces[0];
      }

      stream->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
      count++;
    }
    if(count == 0)
      throw ProblemSetupException("Variable: "+stream->name+" does not have any initial value regions",
                                  __FILE__, __LINE__);

    setup->registerTransportedVariable(mymatls,
                                       stream->massFraction_CCLabel,
                                       stream->massFraction_source_CCLabel);
    streams.push_back(stream);
  }
  if(streams.size() == 0)
    throw ProblemSetupException("Mixing specified with no streams!", __FILE__, __LINE__);

  for (ProblemSpecP child = params->findBlock("reaction"); child != 0;
       child = child->findNextBlock("reaction")) {
    Reaction* rxn = scinew Reaction();
    string from;
    child->require("from", from);
    vector<Stream*>::iterator iter = streams.begin();
    for(;iter != streams.end(); iter++)
      if((*iter)->name == from)
        break;
    if(iter == streams.end())
      throw ProblemSetupException("Reaction needs stream: "+from+" but not found", __FILE__, __LINE__);

    string to;
    child->require("to", to);
    for(;iter != streams.end(); iter++)
      if((*iter)->name == to)
        break;
    if(iter == streams.end())
      throw ProblemSetupException("Reaction needs stream: "+to+" but not found", __FILE__, __LINE__);
    rxn->fromStream = (*iter)->index;
    rxn->toStream = (*iter)->index;

    child->require("energyRelease", rxn->energyRelease);
    child->require("rate", rxn->rate);

    reactions.push_back(rxn);
  }
  if(reactions.size() > 1)
    throw ProblemSetupException("More than one reaction not finished", __FILE__, __LINE__);

//  setup.registerMixer(matl);
}

void Mixing::scheduleInitialize(SchedulerP& sched,
                                const LevelP& level,
                                const ModelInfo*)
{
  Task* t = scinew Task("Mixing::initialize",
                        this, &Mixing::initialize);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->computes(stream->massFraction_CCLabel);
  }
  sched->addTask(t, level->eachPatch(), mymatls);
}

void Mixing::initialize(const ProcessorGroup*, 
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*,
                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      CCVariable<double> sum;
      new_dw->allocateTemporary(sum, patch);
      sum.initialize(0);
      for(vector<Stream*>::iterator iter = streams.begin();
          iter != streams.end(); iter++){
        Stream* stream = *iter;
        CCVariable<double> mf;
        new_dw->allocateAndPut(mf, stream->massFraction_CCLabel, matl, patch);
        mf.initialize(0);
        for(vector<Region*>::iterator iter = stream->regions.begin();
            iter != stream->regions.end(); iter++){
          Region* region = *iter;
          //Box b1 = region->piece->getBoundingBox();
          //Box b2 = patch->getBox();
          //Box b = b1.intersect(b2);
   
          for(CellIterator iter = patch->getExtraCellIterator();
              !iter.done(); iter++){
 
            Point p = patch->cellPosition(*iter);
            if(region->piece->inside(p))
              mf[*iter] = region->initialMassFraction;
          } // Over cells
        } // Over regions
        for(CellIterator iter = patch->getExtraCellIterator();
            !iter.done(); iter++)
          sum[*iter] += mf[*iter];
      } // Over streams
      for(CellIterator iter = patch->getExtraCellIterator();
          !iter.done(); iter++){
        if(sum[*iter] != 1.0){
          ostringstream msg;
          msg << "Initial massFraction != 1.0: value=";
          msg << sum[*iter] << " at " << *iter;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
      } // Over cells
    } // Over matls
  }
}
      
void Mixing::scheduleComputeStableTimestep(SchedulerP&,
                                           const LevelP&,
                                           const ModelInfo*)
{
  // None necessary...
}
      

void Mixing::scheduleComputeModelSources(SchedulerP& sched,
                                              const LevelP& level,
                                              const ModelInfo* mi)
{
  Task* t = scinew Task("Mixing::computeModelSources",this, 
                        &Mixing::computeModelSources, mi);
  t->modifies(mi->modelEng_srcLabel);
  t->requires(Task::OldDW, mi->rho_CCLabel, Ghost::None);
  t->requires(Task::OldDW, mi->delT_Label,  level.get_rep());
  
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->requires(Task::OldDW, stream->massFraction_CCLabel, Ghost::None);
    t->modifies(stream->massFraction_source_CCLabel);
  }

  sched->addTask(t, level->eachPatch(), mymatls);
}

void Mixing::computeModelSources(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw,
                                   const ModelInfo* mi)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);

      for(vector<Reaction*>::iterator iter = reactions.begin();
          iter != reactions.end(); iter++){
        Reaction* rxn = *iter;
        Stream* from = streams[rxn->fromStream];
        constCCVariable<double> from_mf;
        old_dw->get(from_mf, from->massFraction_CCLabel, matl, patch, Ghost::None, 0);
        Stream* to = streams[rxn->toStream];
        constCCVariable<double> to_mf;
        old_dw->get(to_mf, to->massFraction_CCLabel, matl, patch, Ghost::None, 0);

        constCCVariable<double> density;
        old_dw->get(density, mi->rho_CCLabel, matl, patch, Ghost::None, 0);

        Vector dx = patch->dCell();
        double volume = dx.x()*dx.y()*dx.z();

        delt_vartype delT;
        old_dw->get(delT, mi->delT_Label,getLevel(patches) );
        double dt = delT;

        CCVariable<double> energySource;
        new_dw->getModifiable(energySource,   mi->modelEng_srcLabel,
                              matl, patch);

        CCVariable<double> mass_source_from;
        new_dw->getModifiable(mass_source_from, from->massFraction_source_CCLabel,
                              matl, patch);
        CCVariable<double> mass_source_to;
        new_dw->getModifiable(mass_source_to, to->massFraction_source_CCLabel,
                              matl, patch);

        double max = dt*rxn->rate;
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
          IntVector idx = *iter;
          double mass = density[idx]*volume;
          double moles_from = from_mf[idx]*mass/from->props.molecularWeight;
          double moles_rxn = Min(moles_from, max);
          double release = moles_rxn * rxn->energyRelease;
          cerr << "idx=" << idx << ", moles_rxn=" << moles_rxn << ", release=" << release << '\n';
          // Convert energy to temperature...
          energySource[idx] += release;

          double mass_rxn = moles_rxn*to->props.molecularWeight;

          // Modify the mass fractions
          mass_source_from[idx] -= mass_rxn;
          mass_source_to[idx] += mass_rxn;
        }
      }
    }
  }
}


//______________________________________________________________________
void Mixing::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                     const LevelP&,
                                                     const MaterialSet*)
{
  // do nothing      
}
void Mixing::computeSpecificHeat(CCVariable<double>&,
                                 const Patch*,
                                 DataWarehouse*,
                                 const int)
{
}
//______________________________________________________________________
//
void Mixing::scheduleErrorEstimate(const LevelP&,
                                   SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void Mixing::scheduleTestConservation(SchedulerP&,
                                      const PatchSet*,
                                      const ModelInfo*)
{
  // Not implemented yet
}
