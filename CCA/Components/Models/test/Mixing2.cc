
#include <Packages/Uintah/CCA/Components/Models/test/Mixing2.h>
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
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>
#include <cantera/zerodim.h>
#include <stdio.h>


using namespace Uintah;
using namespace std;

Mixing2::Mixing2(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  gas = 0;
  reactor = 0;
}

Mixing2::~Mixing2()
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
  delete gas;
  delete reactor;
}

Mixing2::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("massFraction", initialMassFraction);
}

void Mixing2::problemSetup(GridP&, SimulationStateP& in_state,
			   ModelSetup* setup)
{
  sharedState = in_state;
  matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = matl->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();

  // determine the specific heat of that matl.
  Material* matl = sharedState->getMaterial( m[0] );
  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
  if (ice_matl){
    d_cv = ice_matl->getSpecificHeat();
  }   

  // Parse the Cantera XML file
  string fname;
  params->get("file", fname);
  string id;
  params->get("id", id);
  try {
    gas = new IdealGasMix(fname, id);
    int nsp = gas->nSpecies();
    int nel = gas->nElements();
#if 0
    cerr << "refPressure=" << gas->refPressure() << '\n';
    gas->setState_TPY(200, 101325, "H2:1");
#endif
    if(d_myworld->myrank() == 0){
      cerr.precision(17);
      cerr << "Using ideal gas " << id << "(from " << fname << ") with " << nel << " elements and " << nsp << " species\n";
      gas->setState_TPY(300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
      cerr << *gas;
#if 0
      
      //equilibrate(*gas, UV);
      gas->setState_TPY(300, 101325, "H2:1");
      cerr << *gas;
#endif
    }
    for (int k = 0; k < nsp; k++) {
      Stream* stream = new Stream();
      stream->index = k;
      stream->name = gas->speciesName(k);
      string mfname = "massFraction-"+stream->name;
      stream->massFraction_CCLabel = VarLabel::create(mfname, CCVariable<double>::getTypeDescription());
      string mfsname = "massFractionSource-"+stream->name;
      stream->massFraction_source_CCLabel = VarLabel::create(mfsname, CCVariable<double>::getTypeDescription());
      
      setup->registerTransportedVariable(mymatls->getSubset(0),
					 stream->massFraction_CCLabel,
					 stream->massFraction_source_CCLabel);
      streams.push_back(stream);
      names[stream->name] = stream;
    }

  }
  catch (CanteraError) {
    showErrors(cerr);
    cerr << "test failed." << endl;
    throw InternalError("Cantera failed");
  }

  if(streams.size() == 0)
    throw ProblemSetupException("Mixing2 specified with no streams!");

  for (ProblemSpecP child = params->findBlock("stream"); child != 0;
       child = child->findNextBlock("stream")) {
    string name;
    child->getAttribute("name", name);
    map<string, Stream*>::iterator iter = names.find(name);
    if(iter == names.end())
      throw ProblemSetupException("Stream "+name+" species not found");
    Stream* stream = iter->second;
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
    }
    if(stream->regions.size() == 0)
      throw ProblemSetupException("Variable: "+stream->name+" does not have any initial value regions");

  }
#if 0
  try {
    IdealGasMix* gas = new IdealGasMix("gri30.xml", "gri30");
    IdealGasMix* gas2 = new IdealGasMix("gri30.xml", "gri30");
    gas->setState_TPY(1300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
    gas2->setState_TPY(1300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
    cerr << *gas;
    cerr << *gas2;
    Reactor r;
  
    // specify the thermodynamic property and kinetics managers
    r.setThermoMgr(*gas);
    r.setKineticsMgr(*gas);

    Reactor r2;
    Reservoir env;
  
    // specify the thermodynamic property and kinetics managers
    r2.setThermoMgr(*gas2);
    r2.setKineticsMgr(*gas2);
    env.setThermoMgr(*gas2);

    // create a flexible, insulating wall between the reactor and the
    // environment
    Wall w;
    w.install(r2,env);

    // set the "Vdot coefficient" to a large value, in order to
    // approach the constant-pressure limit; see the documentation 
    // for class Reactor
    w.setExpansionRateCoeff(1.e9);
    w.setArea(1.0);

    double dt=1e-5;
    double max = 1.e-1;
    ofstream out1("press0.dat");
    ofstream out2("temp0.dat");
    for(double t = 0;t<max;t+=dt){
      if(t != 0)
	r.advance(t);
      if(t != 0)
	r2.advance(t);
      out1 << t << " " << gas->pressure() << " " << gas2->pressure() << '\n';
      out2 << t << " " << gas->temperature() << " " << gas2->temperature() << '\n';
    }
  }
  // handle exceptions thrown by Cantera
  catch (CanteraError) {
    showErrors(cout);
    cout << " terminating... " << endl;
  }
#endif
}

void Mixing2::scheduleInitialize(SchedulerP& sched,
				const LevelP& level,
				const ModelInfo*)
{
  Task* t = scinew Task("Mixing2::initialize",
			this, &Mixing2::initialize);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->computes(stream->massFraction_CCLabel);
  }
  sched->addTask(t, level->eachPatch(), mymatls);
}

void Mixing2::initialize(const ProcessorGroup*, 
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
      
void Mixing2::scheduleComputeStableTimestep(SchedulerP&,
					   const LevelP&,
					   const ModelInfo*)
{
  // None necessary...
}
      
void Mixing2::scheduleMassExchange(SchedulerP& sched,
				  const LevelP& level,
				  const ModelInfo* mi)
{
  // None required
}

void Mixing2::scheduleMomentumAndEnergyExchange(SchedulerP& sched,
					       const LevelP& level,
					       const ModelInfo* mi)
{
  Task* t = scinew Task("Mixing2::react",
			this, &Mixing2::react, mi);
  t->modifies(mi->energy_source_CCLabel);
  t->requires(Task::OldDW, mi->density_CCLabel, Ghost::None);
  t->requires(Task::OldDW, mi->pressure_CCLabel, Ghost::None);
  t->requires(Task::OldDW, mi->temperature_CCLabel, Ghost::None);
  t->requires(Task::OldDW, mi->delT_Label);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->requires(Task::OldDW, stream->massFraction_CCLabel, Ghost::None);
    t->modifies(stream->massFraction_source_CCLabel);
  }

  sched->addTask(t, level->eachPatch(), mymatls);
}

void Mixing2::react(const ProcessorGroup*, 
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

      constCCVariable<double> density;
      old_dw->get(density, mi->density_CCLabel, matl, patch, Ghost::None, 0);
      constCCVariable<double> pressure;
      old_dw->get(pressure, mi->pressure_CCLabel, matl, patch, Ghost::None, 0);
      constCCVariable<double> temperature;
      old_dw->get(temperature, mi->temperature_CCLabel, matl, patch, Ghost::None, 0);

      CCVariable<double> energySource;
      new_dw->getModifiable(energySource,   mi->energy_source_CCLabel,
			    matl, patch);

      Vector dx = patch->dCell();
      double volume = dx.x()*dx.y()*dx.z();

      delt_vartype delT;
      old_dw->get(delT, mi->delT_Label);
      double dt = delT;

      int numSpecies = streams.size();
      StaticArray<constCCVariable<double> > mf(numSpecies);
      StaticArray<CCVariable<double> > mfsource(numSpecies);
      int index = 0;
      double* tmp_mf = new double[numSpecies];
      double* new_mf = new double[numSpecies];
      for(vector<Stream*>::iterator iter = streams.begin();
	  iter != streams.end(); iter++, index++){
	Stream* stream = *iter;
	constCCVariable<double> species_mf;
	old_dw->get(species_mf, stream->massFraction_CCLabel, matl, patch, Ghost::None, 0);
	mf[index] = species_mf;

	new_dw->allocateAndPut(mfsource[index], stream->massFraction_source_CCLabel,
			       matl, patch, Ghost::None, 0);
      }

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	IntVector idx = *iter;
	double mass = density[idx]*volume;
	
	for(int i = 0; i< numSpecies; i++)
	  tmp_mf[i] = mf[i][*iter];

#if 1
	double sum = 0;
	for(int i=0;i<numSpecies;i++){
	  //ASSERT(tmp_mf[i] >= 0 && tmp_mf[i] <= 1);
	  if(tmp_mf[i] < -1.e-8)
	    cerr << "mf[" << i << "]=" << tmp_mf[i] << '\n';
	  if(tmp_mf[i] > 1+1.e-8)
	    cerr << "mf[" << i << "]=" << tmp_mf[i] << '\n';
	  sum += tmp_mf[i];
	}
	if(sum < 1-1.e-8 || sum > 1+1.e-8){
	  cerr << "mf sum" << idx << "=" << sum << '\n';
#if 0
	  for(int i=0;i<numSpecies;i++)
	    cerr << i << ": " << tmp_mf[i] << " ";
	  cerr << '\n';
#endif
	}
#endif
#if 0
	for(int i=0;i<numSpecies;i++){
	  if(Abs(tmp_mf[i]) > 1.e-10)
	     cerr << "b: " << gas->speciesName(i) << " = " << tmp_mf[i] << '\n';
	}
#endif
	
	double temp = temperature[*iter];
	double press = pressure[*iter];
	gas->setState_TPY(temp, press, tmp_mf);
#if 0
	cerr << "Cp=" << gas->cp_mass() << '\n';
	cerr << "Cv=" << gas->cv_mass() << '\n';
	cerr << "gamma=" << gas->cp_mass()/gas->cv_mass() << '\n';
#endif
	
	// create a reactor
	Reactor r;
	  
	// specify the thermodynamic property and kinetics managers
	r.setThermoMgr(*gas);
	r.setKineticsMgr(*gas);

#if 0
	double orig_density = gas->density();
	double orig_enthalpy = gas->enthalpy_mass();
	double orig_energy = gas->intEnergy_mass();
	cerr << "ICE density: " << density[idx] << '\n';
	cerr << "enth=" << gas->enthalpy_mass()*mass << ", volume=" << volume << '\n';
	cerr << "eng=" << gas->intEnergy_mass()*mass << ", volume=" << volume << '\n';
	cerr << "b" << *iter << ": " << "\t" << gas->temperature() << "\t" << gas->density() << "\t" << gas->pressure() << '\n';
#endif
	r.advance(dt);
#if 0
	cerr << "a" << *iter << ": " << "\t" << gas->temperature() << "\t" << gas->density() << "\t" << gas->pressure() << '\n';
	double dpress = gas->pressure()-press;
#endif
	double dtemp = gas->temperature()-temp;
#if 0
	double denthalpy = gas->enthalpy_mass()-orig_enthalpy;
	double denergy = gas->intEnergy_mass()-orig_energy;
#endif
	double energyx = dtemp*d_cv*mass;
	energySource[idx] += energyx;
	gas->getMassFractions(new_mf);
#if 0
	double ddens = gas->density()-orig_density;
	cerr << "dpress=" << dpress << ", dtemp=" << dtemp << ", ddens=" << ddens << ", denthalpy=" << denthalpy << ", denergy=" << denergy << '\n';
	double dt1 = dtemp * 1293.908298567086* 1.240786866259094 * mass;
	double dt2 = energyx/(mass*1293.908298567086* 1.240786866259094);
	cerr << "denergy1=" << dt1 << ", dtemp2=" << dt2 << '\n';
	for(int i=0;i<numSpecies;i++){
	  double diff = new_mf[i] - tmp_mf[i];
	  if(Abs(diff) > 1.e-10)
	    cerr << gas->speciesName(i) << " += " << diff << '\n';
	}
	for(int i=0;i<numSpecies;i++){
	  if(Abs(new_mf[i]) > 1.e-10)
	     cerr << "a: " << gas->speciesName(i) << " = " << new_mf[i] << '\n';
	}
	cerr << "eng=" << gas->enthalpy_mass()*mass << ", volume=" << volume << '\n';
#endif
#if 0
	if(idx == IntVector(2,2,0)){
	  static ofstream out1("press.dat");
	  static ofstream out2("temp.dat");
	  out1 << sharedState->getElapsedTime() << " " << gas->pressure() << " " << press << '\n';
	  out2 << sharedState->getElapsedTime() << " " << gas->temperature() << " " << " " << temp << '\n';
	}
#endif
	for(int i = 0; i< numSpecies; i++)
	  mfsource[i][*iter] += new_mf[i]-tmp_mf[i];
#if 0
	// Modify the mass fractions
	mass_source_from[idx] -= mass_rxn;
	mass_source_to[idx] += mass_rxn;
#endif
      }
      {
	static ofstream outt("temp.dat");
	outt << sharedState->getElapsedTime() << " " << temperature[IntVector(2,2,0)] << " " << temperature[IntVector(3,2,0)] << " " << temperature[IntVector(3,3,0)] << '\n';
	outt.flush();
	static ofstream outp("press.dat");
	outp << sharedState->getElapsedTime() << " " << pressure[IntVector(2,2,0)] << " " << pressure[IntVector(3,2,0)] << " " << pressure[IntVector(3,3,0)] << '\n';
	outp.flush();
      }
      delete[] tmp_mf;
      delete[] new_mf;
    }
  }
}

