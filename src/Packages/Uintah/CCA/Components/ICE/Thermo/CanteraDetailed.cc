
#include <Packages/Uintah/CCA/Components/ICE/Thermo/CanteraDetailed.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Util/ProgressiveWarning.h>
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>
#include <cantera/zerodim.h>

using namespace Uintah;

CanteraDetailed::CanteraDetailed(ProblemSpecP& ps, ModelSetup* setup,
                                 ICEMaterial* ice_matl)
  : ThermoInterface(ice_matl), lb(ice_matl->getLabel())
{
  vector<int> m(1);
  m[0] = ice_matl->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();

  ps->require("thermal_conductivity", d_thermalConductivity);
  // Parse the Cantera XML file
  string fname;
  ps->require("file", fname);
  string id;
  ps->require("id", id);
  try {
    d_gas = new IdealGasMix(fname, id);
    int nsp = d_gas->nSpecies();
    int nel = d_gas->nElements();
    cerr << "Using ideal gas " << id << "(from " << fname << ") with " << nel << " elements and " << nsp << " species\n";
    d_gas->setState_TPY(300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
  } catch (CanteraError) {
    showErrors(cerr);
    throw InternalError("Cantera initialization failed", __FILE__, __LINE__);
  }
  int nsp = d_gas->nSpecies();
  for (int k = 0; k < nsp; k++) {
    Stream* stream = new Stream();
    stream->index = k;
    stream->name = d_gas->speciesName(k);
    string mfname = "massFraction-"+stream->name;
    stream->massFraction_CCLabel = VarLabel::create(mfname, CCVariable<double>::getTypeDescription());
    string mflname = "massFraction-reacted-"+stream->name;
    stream->massFraction_reacted_CCLabel = VarLabel::create(mflname, CCVariable<double>::getTypeDescription());
    
    setup->registerTransportedVariable(mymatls->getSubset(0),
                                       Task::NewDW,
                                       stream->massFraction_reacted_CCLabel,
                                       stream->massFraction_CCLabel,
                                       0);
    streams.push_back(stream);
    names[stream->name] = stream;
  }
  for (ProblemSpecP child = ps->findBlock("stream"); child != 0;
       child = child->findNextBlock("stream")) {
    string name;
    child->getAttribute("name", name);
    map<string, Stream*>::iterator iter = names.find(name);
    if(iter == names.end())
      throw ProblemSetupException("Stream "+name+" species not found", __FILE__, __LINE__);
    Stream* stream = iter->second;
    for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
	 geom_obj_ps != 0;
	 geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
      
      vector<GeometryPiece*> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);
      
      GeometryPiece* mainpiece;
      if(pieces.size() == 0){
	throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
      } else if(pieces.size() > 1){
	mainpiece = scinew UnionGeometryPiece(pieces);
      } else {
	mainpiece = pieces[0];
      }

      stream->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
    }
    if(stream->regions.size() == 0)
      throw ProblemSetupException("Variable: "+stream->name+" does not have any initial value regions",
                                  __FILE__, __LINE__);

  }
}

CanteraDetailed::~CanteraDetailed()
{
  delete d_gas;
}

void CanteraDetailed::scheduleInitializeThermo(SchedulerP& sched,
                                               const PatchSet* patches)
{
  Task* t = scinew Task("CanteraDetailed::initialize",
			this, &CanteraDetailed::initialize);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->computes(stream->massFraction_CCLabel);
  }
  sched->addTask(t, patches, mymatls);
}

CanteraDetailed::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("massFraction", initialMassFraction);
}

void CanteraDetailed::initialize(const ProcessorGroup*, 
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
	    if(region->piece->inside(p)){
	      mf[*iter] = region->initialMassFraction;
            }
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
      
void CanteraDetailed::scheduleReactions(SchedulerP& sched,
                                        const PatchSet* patches)
{
  Task* t = scinew Task("CanteraDetailed::react",
			this, &CanteraDetailed::react);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->requires(Task::OldDW, stream->massFraction_CCLabel, Ghost::None, 0);
    t->computes(stream->massFraction_reacted_CCLabel);
  }
  t->requires(Task::OldDW, lb->int_eng_CCLabel, Ghost::None);
  t->requires(Task::OldDW, lb->sp_vol_CCLabel, Ghost::None);
  sched->addTask(t, patches, mymatls);
}

void CanteraDetailed::react(const ProcessorGroup*, 
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
  
      int numSpecies = streams.size();
      StaticArray<constCCVariable<double> > mf(numSpecies);
      StaticArray<CCVariable<double> > mfreacted(numSpecies);
      int index = 0;
      double* tmp_mf = new double[numSpecies];
      double* new_mf = new double[numSpecies];
      for(vector<Stream*>::iterator iter = streams.begin();
	  iter != streams.end(); iter++, index++){
	Stream* stream = *iter;
	constCCVariable<double> species_mf;
	old_dw->get(species_mf, stream->massFraction_CCLabel, matl, patch, Ghost::None, 0);
	mf[index] = species_mf;

	new_dw->allocateAndPut(mfreacted[index], stream->massFraction_reacted_CCLabel,
			       matl, patch, Ghost::None, 0);
      }
      constCCVariable<double> int_eng;
      old_dw->get(int_eng, lb->int_eng_CCLabel, matl, patch, Ghost::None, 0);
      constCCVariable<double> spvol;
      old_dw->get(spvol, lb->sp_vol_CCLabel, matl, patch, Ghost::None, 0);
      delt_vartype delT;
      old_dw->get(delT, lb->delTLabel);
      double dt = delT;

      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
	IntVector idx = *iter;
	
	for(int i = 0; i< numSpecies; i++)
	  tmp_mf[i] = mf[i][*iter];

#if 1
	double sum = 0;
	for(int i=0;i<numSpecies;i++){
	  //ASSERT(tmp_mf[i] >= 0 && tmp_mf[i] <= 1);
	  if(tmp_mf[i] < -1.e-8)
	    cerr << "mf[" << i << ", " << d_gas->speciesName(i) << "]=" << tmp_mf[i] << '\n';
	  if(tmp_mf[i] > 1+1.e-8)
	    cerr << "mf[" << i << ", " << d_gas->speciesName(i) << "]=" << tmp_mf[i] << '\n';
	  sum += tmp_mf[i];
	}
	if(sum < 1-1.e-8 || sum > 1+1.e-8){
	  cerr << "mf sum" << idx << "=" << sum << '\n';
	}
#endif
	
        d_gas->setMassFractions(tmp_mf);
	d_gas->setState_UV(int_eng[*iter], spvol[*iter]);

        //        cerr << iter << ": before: T=" << d_gas->temperature() << " P=" << d_gas->pressure() << "\n";
        double t0 = d_gas->temperature();
	  
	// specify the thermodynamic property and kinetics managers
        Reactor r1;
        r1.setThermoMgr(*d_gas);
        r1.setKineticsMgr(*d_gas);
        ReactorNet sim;
        sim.addReactor(&r1);
	sim.advance(dt);
        //cerr << iter << ": after: T=" << d_gas->temperature() << " P=" << d_gas->pressure() << "\n\n";
#if 1
        double t1 = d_gas->temperature();
        double threshold = 0.10;
        double diff = (t1-t0)/t0;
        if(diff > threshold){
          static ProgressiveWarning warn("More than 10% temp difference in single timestep");
          warn.invoke();
        }
#endif

	d_gas->getMassFractions(new_mf);
	for(int i = 0; i< numSpecies; i++)
	  mfreacted[i][*iter] = new_mf[i];
      }
      delete[] tmp_mf;
      delete[] new_mf;
    }
  }
}

void CanteraDetailed::addTaskDependencies_general(Task* t, State state,
                                                  int numGhostCells)
{
  // State          DW    Label
  // ----------------------
  // Old            Old   unreacted
  // Intermediate   New   reacted
  // New            New   unreacted
  Task::WhichDW fromDW = (state == OldState? Task::OldDW : Task::NewDW);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    t->requires(fromDW, var,
                numGhostCells == 0? Ghost::None : Ghost::AroundCells,
                numGhostCells);
  }
}

void CanteraDetailed::addTaskDependencies_thermalDiffusivity(Task* t, State state,
                                                             int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::addTaskDependencies_thermalConductivity(Task* t, State state,
                                                              int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::addTaskDependencies_cp(Task* t, State state,
                                             int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::addTaskDependencies_cv(Task* t, State state,
                                             int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::addTaskDependencies_gamma(Task* t, State state,
                                                int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::addTaskDependencies_R(Task* t, State state,
                                            int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::addTaskDependencies_Temp(Task* t, State state,
                                               int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::addTaskDependencies_int_eng(Task* t, State state,
                                                  int numGhostCells)
{
  addTaskDependencies_general(t, state, numGhostCells);
}

void CanteraDetailed::compute_thermalDiffusivity(CellIterator iter,
                                                 CCVariable<double>& thermalDiffusivity,
                                                 DataWarehouse* old_dw, DataWarehouse* new_dw,
                                                 State state, const Patch* patch,
                                                 int matl, int numGhostCells,
                                                 const constCCVariable<double>& int_eng,
                                                 const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;!iter.done();iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    thermalDiffusivity[*iter] = d_thermalConductivity/d_gas->cp_mass() * sp_vol[*iter];
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_thermalConductivity(CellIterator iter,
                                                  CCVariable<double>& thermalConductivity,
                                                  DataWarehouse* old_dw, DataWarehouse* new_dw,
                                                  State state, const Patch* patch,
                                                  int matl, int numGhostCells,
                                                  const constCCVariable<double>& int_eng,
                                                  const constCCVariable<double>& sp_vol)
{
  for(;!iter.done();iter++)
    thermalConductivity[*iter] = d_thermalConductivity;
}

void CanteraDetailed::compute_cp(CellIterator iter, CCVariable<double>& cp,
                                 DataWarehouse* old_dw, DataWarehouse* new_dw,
                                 State state, const Patch* patch,
                                 int matl, int numGhostCells,
                                 const constCCVariable<double>& int_eng,
                                 const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;!iter.done();iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cp[*iter] = d_gas->cp_mass();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_cv(CellIterator iter, CCVariable<double>& cv,
                                 DataWarehouse* old_dw, DataWarehouse* new_dw,
                                 State state, const Patch* patch,
                                 int matl, int numGhostCells,
                                 const constCCVariable<double>& int_eng,
                                 const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;!iter.done();iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cv[*iter] = d_gas->cv_mass();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                                    DataWarehouse* old_dw, DataWarehouse* new_dw,
                                    State state, const Patch* patch,
                                    int matl, int numGhostCells,
                                    const constCCVariable<double>& int_eng,
                                    const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;!iter.done();iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    gamma[*iter] = d_gas->cp_mass()/d_gas->cv_mass();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_R(CellIterator iter, CCVariable<double>& R,
                                DataWarehouse* old_dw, DataWarehouse* new_dw,
                                State state, const Patch* patch,
                                int matl, int numGhostCells,
                                const constCCVariable<double>& int_eng,
                                const constCCVariable<double>& sp_vol)
{
  cerr << "csm not done: " << __LINE__ << '\n';
#if 0
  double tmp = (d_gamma-1) * d_specificHeat;
  for(;!iter.done();iter++)
    R[*iter] = tmp;
#endif
}

void CanteraDetailed::compute_Temp(CellIterator iter, CCVariable<double>& temp,
                                   DataWarehouse* old_dw, DataWarehouse* new_dw,
                                   State state, const Patch* patch,
                                   int matl, int numGhostCells,
                                   const constCCVariable<double>& int_eng,
                                   const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;!iter.done();iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    temp[*iter] = d_gas->temperature();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                      DataWarehouse* old_dw, DataWarehouse* new_dw,
                                      State state, const Patch* patch,
                                      int matl, int numGhostCells,
                                      const constCCVariable<double>& Temp,
                                      const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;!iter.done();iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_TR(Temp[*iter], 1.0/sp_vol[*iter]);
    int_eng[*iter] = d_gas->intEnergy_mass();
    //cerr << "initial temp: " << Temp[*iter] << ", energy=" << int_eng[*iter] << '\n';
  }
  delete[] tmp_mf;
}


void CanteraDetailed::compute_cp(cellList::iterator iter, cellList::iterator end,
                                 CCVariable<double>& cp,
                                 DataWarehouse* old_dw, DataWarehouse* new_dw,
                                 State state, const Patch* patch,
                                 int matl, int numGhostCells,
                                 const constCCVariable<double>& int_eng,
                                 const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;iter != end;iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cp[*iter] = d_gas->cp_mass();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_cv(cellList::iterator iter, cellList::iterator end,
                                 CCVariable<double>& cv,
                                 DataWarehouse* old_dw, DataWarehouse* new_dw,
                                 State state, const Patch* patch,
                                 int matl, int numGhostCells,
                                 const constCCVariable<double>& int_eng,
                                 const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;iter != end;iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cv[*iter] = d_gas->cv_mass();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_gamma(cellList::iterator iter, cellList::iterator end,
                                    CCVariable<double>& gamma,
                                    DataWarehouse* old_dw, DataWarehouse* new_dw,
                                    State state, const Patch* patch,
                                    int matl, int numGhostCells,
                                    const constCCVariable<double>& int_eng,
                                    const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;iter != end;iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    gamma[*iter] = d_gas->cp_mass()/d_gas->cv_mass();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_R(cellList::iterator iter, cellList::iterator end,
                                CCVariable<double>& R,
                                DataWarehouse* old_dw, DataWarehouse* new_dw,
                                State state, const Patch* patch,
                                int matl, int numGhostCells,
                                const constCCVariable<double>& int_eng,
                                const constCCVariable<double>& sp_vol)
{
  cerr << "csm not done: " << __LINE__ << '\n';
#if 0
  double tmp = (d_gamma-1) * d_specificHeat;
  for(;iter != end;iter++)
    R[*iter] = tmp;
#endif
}

void CanteraDetailed::compute_Temp(cellList::iterator iter, cellList::iterator end,
                                   CCVariable<double>& temp,
                                   DataWarehouse* old_dw, DataWarehouse* new_dw,
                                   State state, const Patch* patch,
                                   int matl, int numGhostCells,
                                   const constCCVariable<double>& int_eng,
                                   const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;iter != end;iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    temp[*iter] = d_gas->temperature();
  }
  delete[] tmp_mf;
}

void CanteraDetailed::compute_int_eng(cellList::iterator iter, cellList::iterator end,
                                      CCVariable<double>& int_eng,
                                      DataWarehouse* old_dw, DataWarehouse* new_dw,
                                      State state, const Patch* patch,
                                      int matl, int numGhostCells,
                                      const constCCVariable<double>& Temp,
                                      const constCCVariable<double>& sp_vol)
{
  int numSpecies = streams.size();
  StaticArray<constCCVariable<double> > mf(numSpecies);
  int index = 0;
  double* tmp_mf = new double[numSpecies];
  DataWarehouse* dw = (state == OldState? old_dw : new_dw);
  for(vector<Stream*>::iterator siter = streams.begin();
      siter != streams.end(); siter++, index++){
    Stream* stream = *siter;
    constCCVariable<double> species_mf;
    VarLabel* var = (state == IntermediateState?
                     stream->massFraction_reacted_CCLabel :
                     stream->massFraction_CCLabel);
    dw->get(species_mf, var, matl, patch,
            numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
    mf[index] = species_mf;
  }

  for(;iter != end;iter++){
    for(int i = 0; i< numSpecies; i++)
      tmp_mf[i] = mf[i][*iter];
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_TR(Temp[*iter], 1.0/sp_vol[*iter]);
    int_eng[*iter] = d_gas->intEnergy_mass();
    //cerr << "initial temp: " << Temp[*iter] << ", energy=" << int_eng[*iter] << '\n';
  }
  delete[] tmp_mf;
}
