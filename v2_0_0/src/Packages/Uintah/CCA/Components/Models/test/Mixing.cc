
#include <Packages/Uintah/CCA/Components/Models/test/Mixing.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
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
      delete region->piece;
      delete region;
    }
  }
}

Mixing::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("massFraction", initialMassFraction);
}

void Mixing::problemSetup(GridP&, SimulationStateP& sharedState,
			  ModelSetup* setup)
{
  matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = matl->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();

  // Parse the streams
  int index = 0;
  for (ProblemSpecP child = params->findBlock("stream"); child != 0;
       child = child->findNextBlock("stream")) {
    Stream* stream = new Stream();
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
      
      vector<GeometryPiece*> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);
      
      GeometryPiece* mainpiece;
      if(pieces.size() == 0){
	throw ParameterNotFound("No piece specified in geom_object");
      } else if(pieces.size() > 1){
	mainpiece = scinew UnionGeometryPiece(pieces);
      } else {
	mainpiece = pieces[0];
      }

      stream->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
      count++;
    }
    if(count == 0)
      throw ProblemSetupException("Variable: "+stream->name+" does not have any initial value regions");

    setup->registerTransportedVariable(mymatls->getSubset(0),
				       stream->massFraction_CCLabel,
				       stream->massFraction_source_CCLabel);
    streams.push_back(stream);
  }
  if(streams.size() == 0)
    throw ProblemSetupException("Mixing specified with no streams!");

  for (ProblemSpecP child = params->findBlock("reaction"); child != 0;
       child = child->findNextBlock("reaction")) {
    Reaction* rxn = new Reaction();
    string from;
    child->require("from", from);
    vector<Stream*>::iterator iter = streams.begin();
    for(;iter != streams.end(); iter++)
      if((*iter)->name == from)
	break;
    if(iter == streams.end())
      throw ProblemSetupException("Reaction needs stream: "+from+" but not found");

    string to;
    child->require("to", to);
    for(;iter != streams.end(); iter++)
      if((*iter)->name == to)
	break;
    if(iter == streams.end())
      throw ProblemSetupException("Reaction needs stream: "+to+" but not found");
    rxn->fromStream = (*iter)->index;
    rxn->toStream = (*iter)->index;

    child->require("energyRelease", rxn->energyRelease);
    child->require("rate", rxn->rate);

    reactions.push_back(rxn);
  }
  if(reactions.size() > 1)
    throw ProblemSetupException("More than one reaction not finished");

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
	  Box b1 = region->piece->getBoundingBox();
	  Box b2 = patch->getBox();
	  Box b = b1.intersect(b2);
   
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
	  throw ProblemSetupException(msg.str());
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
      
void Mixing::scheduleMassExchange(SchedulerP& sched,
				  const LevelP& level,
				  const ModelInfo* mi)
{
  // None required
}

void Mixing::scheduleMomentumAndEnergyExchange(SchedulerP& sched,
					       const LevelP& level,
					       const ModelInfo* mi)
{
  Task* t = scinew Task("Mixing::react",
			this, &Mixing::react, mi);
  t->modifies(mi->energy_source_CCLabel);
  t->requires(Task::OldDW, mi->density_CCLabel, Ghost::None);
  t->requires(Task::OldDW, mi->delT_Label);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->requires(Task::OldDW, stream->massFraction_CCLabel, Ghost::None);
    t->modifies(stream->massFraction_source_CCLabel);
  }

  sched->addTask(t, level->eachPatch(), mymatls);
}

void Mixing::react(const ProcessorGroup*, 
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
	old_dw->get(density, mi->density_CCLabel, matl, patch, Ghost::None, 0);

	Vector dx = patch->dCell();
	double volume = dx.x()*dx.y()*dx.z();

	delt_vartype delT;
	old_dw->get(delT, mi->delT_Label);
	double dt = delT;

	CCVariable<double> energySource;
	new_dw->getModifiable(energySource,   mi->energy_source_CCLabel,
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

