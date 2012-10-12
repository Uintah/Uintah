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


#include <CCA/Components/Models/FluidsBased/Mixing2.h>
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
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>
#include <cantera/zerodim.h>
#include <cstdio>

// TODO:
// SGI build
// Memory leaks in cantera
// bigger problems
// profile

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
  mymatls = scinew MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();   

  // Parse the Cantera XML file
  string fname;
  params->get("file", fname);
  string id;
  params->get("id", id);
  try {
    gas = scinew IdealGasMix(fname, id);
    int nsp = gas->nSpecies();
#if 0
    cerr << "refPressure=" << gas->refPressure() << '\n';
    gas->setState_TPY(200, 101325, "H2:1");
#endif
    if(d_myworld->myrank() == 0){
#if 0
      int nel = gas->nElements();
      cerr.precision(17);
      cerr << "Using ideal gas " << id << "(from " << fname << ") with " << nel << " elements and " << nsp << " species\n";
      gas->setState_TPY(300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
      cerr << *gas;
#endif
#if 0
      
      //equilibrate(*gas, UV);
      gas->setState_TPY(300, 101325, "H2:1");
      cerr << *gas;
#endif
    }
    for (int k = 0; k < nsp; k++) {
      Stream* stream = scinew Stream();
      stream->index = k;
      stream->name = gas->speciesName(k);
      string mfname = "massFraction-"+stream->name;
      stream->massFraction_CCLabel = VarLabel::create(mfname, CCVariable<double>::getTypeDescription());
      string mfsname = "massFractionSource-"+stream->name;
      stream->massFraction_source_CCLabel = VarLabel::create(mfsname, CCVariable<double>::getTypeDescription());
      
      setup->registerTransportedVariable(mymatls,
                                         stream->massFraction_CCLabel,
                                         stream->massFraction_source_CCLabel);
      streams.push_back(stream);
      names[stream->name] = stream;
    }

  }
  catch (CanteraError) {
    showErrors(cerr);
    cerr << "test failed." << endl;
    throw InternalError("Cantera failed", __FILE__, __LINE__);
  }

  if(streams.size() == 0)
    throw ProblemSetupException("Mixing2 specified with no streams!", __FILE__, __LINE__);

  for (ProblemSpecP child = params->findBlock("stream"); child != 0;
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
#if 0
  try {
    IdealGasMix* gas = scinew IdealGasMix("gri30.xml", "gri30");
    IdealGasMix* gas2 = scinew IdealGasMix("gri30.xml", "gri30");
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
          throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
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
      

void Mixing2::scheduleComputeModelSources(SchedulerP& sched,
                                               const LevelP& level,
                                               const ModelInfo* mi)
{
  Task* t = scinew Task("Mixing2::computeModelSources",this, 
                        &Mixing2::computeModelSources, mi);
  t->modifies(mi->modelEng_srcLabel);
  t->requires(Task::OldDW, mi->rho_CCLabel,        Ghost::None);
  t->requires(Task::OldDW, mi->press_CCLabel,      Ghost::None);
  t->requires(Task::OldDW, mi->temp_CCLabel,       Ghost::None);
  t->requires(Task::NewDW, mi->specific_heatLabel, Ghost::None);
  t->requires(Task::NewDW, mi->gammaLabel,         Ghost::None);
  t->requires(Task::OldDW, mi->delT_Label,         level.get_rep());
  
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->requires(Task::OldDW, stream->massFraction_CCLabel, Ghost::None);
    t->modifies(stream->massFraction_source_CCLabel);
  }

  sched->addTask(t, level->eachPatch(), mymatls);
}

void Mixing2::computeModelSources(const ProcessorGroup*, 
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
      old_dw->get(density,     mi->rho_CCLabel,        matl, patch, Ghost::None, 0);
      constCCVariable<double> pressure;
      old_dw->get(pressure,    mi->press_CCLabel,      matl, patch, Ghost::None, 0);
      constCCVariable<double> temperature;
      old_dw->get(temperature, mi->temp_CCLabel,       matl, patch, Ghost::None, 0);
      constCCVariable<double> gamma;
      old_dw->get(gamma,       mi->gammaLabel,         matl, patch, Ghost::None, 0);      
      constCCVariable<double> cv;
      old_dw->get(cv,          mi->specific_heatLabel, matl, patch, Ghost::None, 0);
        
      CCVariable<double> energySource;
      new_dw->getModifiable(energySource,   mi->modelEng_srcLabel,
                            matl, patch);

      Vector dx = patch->dCell();
      double volume = dx.x()*dx.y()*dx.z();

      delt_vartype delT;
      old_dw->get(delT, mi->delT_Label, getLevel(patches));
      double dt = delT;

      int numSpecies = streams.size();
      StaticArray<constCCVariable<double> > mf(numSpecies);
      StaticArray<CCVariable<double> > mfsource(numSpecies);
      int index = 0;
      double* tmp_mf =scinew double[numSpecies];
      double* new_mf =scinew double[numSpecies];
      for(vector<Stream*>::iterator iter = streams.begin();
          iter != streams.end(); iter++, index++){
        Stream* stream = *iter;
        constCCVariable<double> species_mf;
        old_dw->get(species_mf, stream->massFraction_CCLabel, matl, patch, Ghost::None, 0);
        mf[index] = species_mf;

        new_dw->allocateAndPut(mfsource[index], stream->massFraction_source_CCLabel,
                               matl, patch, Ghost::None, 0);
      }

      Reactor r;
      double etotal = 0;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector idx = *iter;
        double mass = density[idx]*volume;
        
        for(int i = 0; i< numSpecies; i++)
          tmp_mf[i] = mf[i][*iter];

#if 0
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
          
        // specify the thermodynamic property and kinetics managers
        r.setThermoMgr(*gas);
        r.setKineticsMgr(*gas);
        r.initialize();
        r.advance(dt);
        double dtemp = gas->temperature()-temp;
        double energyx = dtemp*cv[*iter]*mass;
        energySource[idx] += energyx;
        etotal += energyx;
        gas->getMassFractions(new_mf);
        for(int i = 0; i< numSpecies; i++)
          mfsource[i][*iter] += new_mf[i]-tmp_mf[i];
      }
      cerr << "Mixing2 total energy: " << etotal << ", release rate=" << etotal/dt << '\n';
#if 0
      {
        static ofstream outt("temp.dat");
        outt << sharedState->getElapsedTime() << " " << temperature[IntVector(2,2,0)] << " " << temperature[IntVector(3,2,0)] << " " << temperature[IntVector(3,3,0)] << '\n';
        outt.flush();
        static ofstream outp("press.dat");
        outp << sharedState->getElapsedTime() << " " << pressure[IntVector(2,2,0)] << " " << pressure[IntVector(3,2,0)] << " " << pressure[IntVector(3,3,0)] << '\n';
        outp.flush();
      }
#endif
      delete[] tmp_mf;
      delete[] new_mf;
    }
  }
}
//______________________________________________________________________
void Mixing2::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                      const LevelP&,
                                                      const MaterialSet*)
{
  // do nothing      
}
//______________________________________________________________________
//
void Mixing2::scheduleErrorEstimate(const LevelP&,
                                    SchedulerP&)
{
  // Not implemented yet
}

