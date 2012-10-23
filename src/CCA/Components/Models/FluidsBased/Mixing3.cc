/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <CCA/Components/Models/FluidsBased/Mixing3.h>
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
#include <cantera/equilibrium.h>
#include <cstdio>

// TODO:
// SGI build
// Memory leaks in cantera
// bigger problems
// profile

using namespace Uintah;
using namespace std;

Mixing3::Mixing3(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  mymatls = 0;
  gas = 0;
  reactor = 0;
  params->require("dtemp", dtemp);
  params->require("dpress", dpress);
  params->require("dmf", dmf);
  params->require("dtfactor", dtfactor);
  nlook=0;
  nmiss=0;
}

Mixing3::~Mixing3()
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

Mixing3::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("massFraction", initialMassFraction);
}

void Mixing3::problemSetup(GridP&, SimulationStateP& in_state,
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
    if(d_myworld->myrank() == 0){
#if 0
      cerr.precision(17);
      cerr << "Using ideal gas " << id << "(from " << fname << ") with " << nel << " elements and " << nsp << " species\n";
      gas->setState_TPY(300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
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
    throw ProblemSetupException("Mixing3 specified with no streams!", __FILE__, __LINE__);

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
}

void Mixing3::scheduleInitialize(SchedulerP& sched,
                                const LevelP& level,
                                const ModelInfo*)
{
  Task* t = scinew Task("Mixing3::initialize",
                        this, &Mixing3::initialize);
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->computes(stream->massFraction_CCLabel);
  }
  sched->addTask(t, level->eachPatch(), mymatls);
}

void Mixing3::initialize(const ProcessorGroup*, 
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
      
void Mixing3::scheduleComputeStableTimestep(SchedulerP&,
                                           const LevelP&,
                                           const ModelInfo*)
{
  // None necessary...
}

void Mixing3::scheduleComputeModelSources(SchedulerP& sched,
                                               const LevelP& level,
                                               const ModelInfo* mi)
{
  Task* t = scinew Task("Mixing3::computeModelSources", this, 
                        &Mixing3::computeModelSources, mi);
  t->modifies(mi->modelEng_srcLabel);
  t->requires(Task::OldDW, mi->rho_CCLabel,   Ghost::None);
  t->requires(Task::OldDW, mi->press_CCLabel, Ghost::None);
  t->requires(Task::OldDW, mi->temp_CCLabel,  Ghost::None);
  t->requires(Task::OldDW, mi->delT_Label,    level.get_rep());
  
  for(vector<Stream*>::iterator iter = streams.begin();
      iter != streams.end(); iter++){
    Stream* stream = *iter;
    t->requires(Task::OldDW, stream->massFraction_CCLabel, Ghost::None);
    t->modifies(stream->massFraction_source_CCLabel);
  }

  sched->addTask(t, level->eachPatch(), mymatls);
}

namespace Uintah {
  struct M3Key {
    M3Key(int nsp, int idt, int itemp, int ipress, int* imf, double dtemp = 0,
          double* mf=0)
      : nsp(nsp), idt(idt), itemp(itemp), ipress(ipress), imf(imf), dtemp(dtemp),
        mf(mf)
    {
      int h = idt + itemp<<3 + ipress<<5;
      for(int i=0;i<nsp;i++)
        h = (h<<1) | (h>>1) | imf[i];
      if(h<0)
        h=-h;
      hash = h;
      next = 0;
    }
    bool operator==(const M3Key& c) const {
      if(idt != c.idt)
        return false;
      if(itemp != c.itemp)
        return false;
      if(ipress != c.ipress)
        return false;
      for(int i=0;i<nsp;i++){
        if(imf[i] != c.imf[i])
          return false;
      }
      return true;
    }

    int nsp;
    int idt;
    int itemp;
    int ipress;
    int* imf;
    int hash;
    double dtemp;
    double* mf;
    M3Key* next;
  };
}

double Mixing3::lookup(int nsp, int idt, int itemp, int ipress, int* imf,
                     double* outmf)
{
  nlook++;
  // Lookup in hash table
  M3Key k(nsp, idt, itemp, ipress, imf);

  // Create it if not found
  M3Key* r = table.lookup(&k);
  if(!r){
    nmiss++;
    double approx_dt = exp(idt/dtfactor);
    int* imfcopy = scinew int[nsp];
    for(int i=0;i<nsp;i++)
      imfcopy[i] = imf[i];
    double* newmf =scinew double[nsp];
    for(int i=0;i<nsp;i++)
      newmf[i] = imf[i]*dmf;
    double temp = itemp*dtemp;
    double press = ipress*dpress;

    cerr << "dt=" << approx_dt << "(" << idt << "), t=" << temp << ", p=" << press << ", mf=";
    for(int i=0;i<nsp;i++){
      if(imf[i])
        cerr << " " << gas->speciesName(i) << ":" << newmf[i];
    }
    cerr << " create\n";

    double dtemp;
    try {
      gas->setState_TPY(temp, press, newmf);
      Reactor r;
      r.setThermoMgr(*gas);
      r.setKineticsMgr(*gas);
      r.initialize();
      r.advance(approx_dt);
      dtemp = gas->temperature()-temp;
      gas->getMassFractions(newmf);
    }   catch (CanteraError) {
      showErrors(cerr);
      cerr << "test failed." << endl;
      throw InternalError("Cantera failed", __FILE__, __LINE__);
    }
    cerr << "After: t=" << gas->temperature() << ", p=" << gas->pressure() << ", mf=";
    for(int i=0;i<nsp;i++){
      if(newmf[i] > dmf)
        cerr << " " << gas->speciesName(i) << ":" << newmf[i];
    }

    r = scinew M3Key(nsp, idt, itemp, ipress, imfcopy, dtemp, newmf);
    table.insert(r);
    double hitrate = (double)nmiss/(double)nlook;
    cerr << "temp: " << temp << " += " << dtemp << ", hitrate=" << hitrate*100 << "%\n";
  }
  for(int i=0;i<nsp;i++)
    outmf[i] = r->mf[i];
  return r->dtemp;
}

void Mixing3::computeModelSources(const ProcessorGroup*, 
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
      constCCVariable<double> pressure;
      constCCVariable<double> temperature;
      constCCVariable<double>cv;
      old_dw->get(density,     mi->rho_CCLabel,        matl, patch, Ghost::None, 0);
      old_dw->get(pressure,    mi->press_CCLabel,      matl, patch, Ghost::None, 0);
      old_dw->get(temperature, mi->temp_CCLabel,       matl, patch, Ghost::None, 0);
      new_dw->get(cv,          mi->specific_heatLabel, matl, patch, Ghost::None, 0);
    
      CCVariable<double> energySource;
      new_dw->getModifiable(energySource, mi->modelEng_srcLabel,matl, patch);

      Vector dx = patch->dCell();
      double volume = dx.x()*dx.y()*dx.z();

      delt_vartype delT;
      old_dw->get(delT, mi->delT_Label, getLevel(patches));
      double dt = delT;

      int numSpecies = streams.size();
      StaticArray<constCCVariable<double> > mf(numSpecies);
      StaticArray<CCVariable<double> > mfsource(numSpecies);
      int index = 0;
      int* imf = scinew int[numSpecies];
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

      double ldt = log(dt)*dtfactor;
      int idt;
      if(ldt > 0)
        idt = (int)ldt;
      else if(ldt == (int)ldt)
        idt = -(int)(-ldt);
      else
        idt = -(int)(-ldt)-1;

      double approx_dt = exp(idt/dtfactor);
      double dtscale = dt/approx_dt;
      if(dtscale < 1 || dtscale > 1.1)
        throw InternalError("Approximation messed!", __FILE__, __LINE__);
      cerr << "dtscale=" << dtscale << '\n';

      double etotal = 0;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector idx = *iter;
        double mass = density[idx]*volume;
        
        for(int i = 0; i< numSpecies; i++){
          tmp_mf[i] = mf[i][*iter];
          imf[i] = (int)(mf[i][*iter]/dmf+0.5);
        }
        
        double temp = temperature[*iter];
        double press = pressure[*iter];
        int itemp = (int)(temp/dtemp+0.5);
        int ipress = (int)(press/dpress+0.5);

        double dtemp = lookup(numSpecies, idt, itemp, ipress, imf, new_mf);
        dtemp *= dtscale;
        double energyx = dtemp * cv[idx] * mass;
        energySource[idx] += energyx;
        etotal += energyx;
        for(int i = 0; i< numSpecies; i++)
          mfsource[i][*iter] += new_mf[i]-tmp_mf[i];
      }
      cerr << "Mixing3 total energy: " << etotal << ", release rate=" << etotal/dt << '\n';
      delete[] tmp_mf;
      delete[] new_mf;
    }
  }
}
//______________________________________________________________________
//
void Mixing3::computeSpecificHeat(CCVariable<double>&,
                                    const Patch*,   
                                    DataWarehouse*, 
                                    const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void Mixing3::scheduleErrorEstimate(const LevelP&,
                                         SchedulerP&)
{
  // Not implemented yet
}
