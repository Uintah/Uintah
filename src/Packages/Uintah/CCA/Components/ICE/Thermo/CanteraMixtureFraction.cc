
#include <Packages/Uintah/CCA/Components/ICE/Thermo/CanteraMixtureFraction.h>
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
#include <cantera/equilibrium.h>

using namespace Uintah;

CanteraMixtureFraction::CanteraMixtureFraction(ProblemSpecP& ps, ModelSetup* setup,
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
  mixtureFraction_CCLabel = VarLabel::create("f",
                                             CCVariable<double>::getTypeDescription());
  setup->registerTransportedVariable(mymatls->getSubset(0),
                                     Task::OldDW,
                                     mixtureFraction_CCLabel,
                                     mixtureFraction_CCLabel,
                                     0);
  for (ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
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
    
    regions.push_back(scinew Region(mainpiece, geom_obj_ps));
  }
  if(regions.size() == 0)
    throw ProblemSetupException("mixtureFraction does not have any initial value regions",
                                  __FILE__, __LINE__);
  
}

CanteraMixtureFraction::~CanteraMixtureFraction()
{
  delete d_gas;
}

void CanteraMixtureFraction::scheduleInitializeThermo(SchedulerP& sched,
                                               const PatchSet* patches)
{
  Task* t = scinew Task("CanteraMixtureFraction::initialize",
			this, &CanteraMixtureFraction::initialize);
  t->computes(mixtureFraction_CCLabel);
  sched->addTask(t, patches, mymatls);
}

CanteraMixtureFraction::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("mixtureFraction", initialMixtureFraction);
}

void CanteraMixtureFraction::initialize(const ProcessorGroup*, 
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      CCVariable<double> f;
      new_dw->allocateAndPut(f, mixtureFraction_CCLabel, matl, patch);
      f.initialize(999);
      for(vector<Region*>::iterator iter = regions.begin();
          iter != regions.end(); iter++){
        Region* region = *iter;
        Box b1 = region->piece->getBoundingBox();
        Box b2 = patch->getBox();
        Box b = b1.intersect(b2);
   
        for(CellIterator iter = patch->getExtraCellIterator();
            !iter.done(); iter++){
          Point p = patch->cellPosition(*iter);
          if(region->piece->inside(p)){
            f[*iter] = region->initialMixtureFraction;
          }
        } // Over cells
      } // Over regions
      for(CellIterator iter = patch->getExtraCellIterator();
          !iter.done(); iter++){
        if(f[*iter] < 0.0 || f[*iter] > 1.0){
          ostringstream msg;
          msg << "Initial massFraction out of range [0,1]: value=";
          msg << f[*iter] << " at " << *iter;
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
        }
      } // Over cells
    } // Over matls
  }
}
      
void CanteraMixtureFraction::scheduleReactions(SchedulerP& sched,
                                        const PatchSet* patches)
{
  // Reactions are implicit - nothing to do here
}

void CanteraMixtureFraction::addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                             int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::addTaskDependencies_thermalConductivity(Task* t, Task::WhichDW dw,
                                                             int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::addTaskDependencies_cp(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::addTaskDependencies_cv(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                               int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                           int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::addTaskDependencies_Temp(Task* t, Task::WhichDW dw,
                                              int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::addTaskDependencies_int_eng(Task* t, Task::WhichDW dw,
                                                 int numGhostCells)
{
  t->requires(Task::OldDW, mixtureFraction_CCLabel,
              numGhostCells == 0? Ghost::None : Ghost::AroundCells,
              numGhostCells);
}

void CanteraMixtureFraction::compute_thermalDiffusivity(CellIterator iter,
                                                 CCVariable<double>& thermalDiffusivity,
                                                 DataWarehouse* new_dw, const Patch* patch,
                                                 int matl, int numGhostCells,
                                                 constCCVariable<double>& int_eng,
                                                 constCCVariable<double>& sp_vol)
{
  cerr << "compute_thermalDiffusivity not finished\n";
}

void CanteraMixtureFraction::compute_thermalConductivity(CellIterator iter,
                                                  CCVariable<double>& thermalConductivity,
                                                  DataWarehouse*, const Patch* patch,
                                                  int matl, int numGhostCells,
                                                  constCCVariable<double>& int_eng)
{
  for(;!iter.done();iter++)
    thermalConductivity[*iter] = d_thermalConductivity;
}

void CanteraMixtureFraction::compute_cp(CellIterator iter, CCVariable<double>& cp,
                                 DataWarehouse* new_dw, const Patch* patch,
                                 int matl, int numGhostCells,
                                 constCCVariable<double>& int_eng)
{
  constCCVariable<double> f;
  new_dw->get(f, mixtureFraction_CCLabel, matl, patch,
              numGhostCells==0?Ghost::None : Ghost::AroundCells, numGhostCells);
  int numSpecies = d_gas->nSpecies();
  double* tmp_mf = new double[numSpecies];

  for(;!iter.done();iter++){
    for(int i = 0; i< numSpecies; i++){
      double mf = f[*iter];
      tmp_mf[i] = mix0[i] * (1-mf) + mix1[i] * mf;
    }
    d_gas->setMassFractions(tmp_mf);
    d_gas->setState_UV(int_eng[*iter], 1.0);
    equilibrate(*d_gas, UV);
    cp[*iter] = d_gas->cp_mass();
  }
  delete[] tmp_mf;
}

void CanteraMixtureFraction::compute_cv(CellIterator iter, CCVariable<double>& cv,
                                 DataWarehouse* new_dw, const Patch* patch,
                                 int matl, int numGhostCells,
                                 constCCVariable<double>& int_eng)
{
  cerr << "compute_cv not finished\n";
}

void CanteraMixtureFraction::compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                                 DataWarehouse* new_dw, const Patch* patch,
                                 int matl, int numGhostCells,
                                 constCCVariable<double>& int_eng)
{
  cerr << "compute_gamma not finished\n";
}

void CanteraMixtureFraction::compute_R(CellIterator iter, CCVariable<double>& R,
                                DataWarehouse*, const Patch* patch,
                                int matl, int numGhostCells,
                                constCCVariable<double>& int_eng)
{
  cerr << "csm not done: " << __LINE__ << '\n';
#if 0
  double tmp = (d_gamma-1) * d_specificHeat;
  for(;!iter.done();iter++)
    R[*iter] = tmp;
#endif
}

void CanteraMixtureFraction::compute_Temp(CellIterator iter, CCVariable<double>& temp,
                                   DataWarehouse* new_dw, const Patch* patch,
                                   int matl, int numGhostCells,
                                   constCCVariable<double>& int_eng)
{
  cerr << "compute_temp not finished\n";
}

void CanteraMixtureFraction::compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                      DataWarehouse* new_dw, const Patch* patch,
                                      int matl, int numGhostCells,
                                      constCCVariable<double>& Temp)
{
  cerr << "compute_int_eng not finished\n";
}
