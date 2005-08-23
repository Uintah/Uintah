
#include <Packages/Uintah/CCA/Components/ICE/Thermo/CanteraDetailed.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>

using namespace Cantera;
using namespace Uintah;

CanteraDetailed::CanteraDetailed(ProblemSpecP& ps)
{
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
    cerr.precision(17);
    cerr << "Using ideal gas " << id << "(from " << fname << ") with " << nel << " elements and " << nsp << " species\n";
    d_gas->setState_TPY(300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
    cerr << *d_gas;
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
    
    cerr << "Cantera setup not finished\n";
#if 0  
    setup->registerTransportedVariable(mymatls->getSubset(0),
                                       stream->massFraction_CCLabel,
                                       0);
#endif
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
                                              const PatchSet* patches,
                                              ICEMaterial* ice_matl)
{
  Task* t = scinew Task("CanteraDetailed::initialize",
			this, &CanteraDetailed::initialize);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->computes(stream->massFraction_CCLabel);
  }
  vector<int> m(1);
  m[0] = ice_matl->getDWIndex();
  MaterialSet* mymatls = new MaterialSet();
  mymatls->addAll(m);
  sched->addTask(t, patches, mymatls);
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
      
void CanteraDetailed::scheduleReactions(SchedulerP& sched,
                                        const PatchSet* patches,
                                        ICEMaterial* ice_matl)
{
  Task* t = scinew Task("CanteraDetailed::initialize",
			this, &CanteraDetailed::react);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->computes(stream->massFraction_CCLabel);
  }
  vector<int> m(1);
  m[0] = ice_matl->getDWIndex();
  MaterialSet* mymatls = new MaterialSet();
  mymatls->addAll(m);
  sched->addTask(t, patches, mymatls);
}

void CanteraDetailed::react(const ProcessorGroup*, 
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse* new_dw)
{
  
}

void CanteraDetailed::addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                            int numGhostCells)
{
  // No additional requirements
}

void CanteraDetailed::addTaskDependencies_thermalConductivity(Task* t, Task::WhichDW dw,
                                                             int numGhostCells)
{
  // No additional requirements
}

void CanteraDetailed::addTaskDependencies_cp(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  // No additional requirements
}

void CanteraDetailed::addTaskDependencies_cv(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  // No additional requirements
}

void CanteraDetailed::addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                               int numGhostCells)
{
  // No additional requirements
}

void CanteraDetailed::addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                           int numGhostCells)
{
  // No additional requirements
}

void CanteraDetailed::addTaskDependencies_Temp(Task* t, Task::WhichDW dw,
                                              int numGhostCells)
{
  // No additional requirements
}

void CanteraDetailed::addTaskDependencies_int_eng(Task* t, Task::WhichDW dw,
                                                 int numGhostCells)
{
  // No additional requirements
}

#if 0
void CanteraDetailed::compute_thermalDiffusivity(CellIterator iter,
                                                      CCVariable<double>& thermalDiffusivity,
                                                      DataWarehouse*,
                                                      constCCVariable<double>& int_eng,
                                                      constCCVariable<double>& sp_vol)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    thermalDiffusivity[*iter] = d_thermalConductivity/d_gas->cp_mass() * sp_vol[*iter];
  }
}

void CanteraDetailed::compute_thermalConductivity(CellIterator iter,
                                                       CCVariable<double>& thermalConductivity,
                                                       DataWarehouse*)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++)
    thermalConductivity[*iter] = d_thermalConductivity;
}

void CanteraDetailed::compute_cp(CellIterator iter, CCVariable<double>& cp,
                                      DataWarehouse*, constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cp[*iter] = d_gas->cp_mass();
  }
}

void CanteraDetailed::compute_cv(CellIterator iter, CCVariable<double>& cv,
                                      DataWarehouse*, constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cv[*iter] = d_gas->cv_mass();
  }
}

void CanteraDetailed::compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                                         DataWarehouse*, constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    gamma[*iter] = d_gas->cp_mass()/d_gas->cv_mass();
  }
}

void CanteraDetailed::compute_R(CellIterator iter, CCVariable<double>& R,
                                     DataWarehouse*, constCCVariable<double>& int_eng)
{
  cerr << "csm not done: " << __LINE__ << '\n';
#if 0
  double tmp = (d_gamma-1) * d_specificHeat;
  for(;!iter.done();iter++)
    R[*iter] = tmp;
#endif
}

void CanteraDetailed::compute_Temp(CellIterator iter, CCVariable<double>& temp,
                                        DataWarehouse*, constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    temp[*iter] = d_gas->temperature();
  }
}

void CanteraDetailed::compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                           DataWarehouse*,
                                           constCCVariable<double>& temp)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_TR(temp[*iter], 1.0);
    int_eng[*iter] = d_gas->intEnergy_mass();
  }
}
#endif
