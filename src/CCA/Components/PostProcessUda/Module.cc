/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/PostProcessUda/Module.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

using namespace Uintah;
using namespace std;

Module::Module()
{
}

Module::Module(ProblemSpecP     & prob_spec,
               MaterialManagerP & materialManager,
               Output           * dataArchiver,
               DataArchive      * dataArchive )
{
  d_materialManager  = materialManager;
  d_dataArchiver = dataArchiver;
  d_dataArchive  = dataArchive;

 if(!d_dataArchiver){
    throw InternalError("Module:couldn't get output port", __FILE__, __LINE__);
  }
}

Module::~Module()
{
}

//______________________________________________________________________
//  Parse the ups file and read timeStart and timeStop
void Module::readTimeStartStop(const ProblemSpecP & ps,
                               double & startTime,
                               double & stopTime)
{
  ProblemSpecP prob_spec = ps;
  prob_spec->require("timeStart",  startTime);
  prob_spec->require("timeStop",   stopTime);

  //__________________________________
  // bulletproofing
  // Start time < stop time
  if(startTime >= stopTime ){
    throw ProblemSetupException("\n ERROR:PostProcess: startTime >= stopTime. \n", __FILE__, __LINE__);
  }
  
  std::vector<int> udaTimesteps;
  d_dataArchive->queryTimesteps( udaTimesteps, d_udaTimes );

  if ( startTime < d_udaTimes[0] ){
    std::ostringstream warn;
    warn << "  ERROR:PostProcess: The startTime (" << startTime
         << ") must be greater than the time at timestep 1 (" << d_udaTimes[0] << ")";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//  Read the ups file and create the material set
//  The material set is the default matl + any optional matls specified on the labels
void Module::createMatlSet(const ProblemSpecP & ps,
                           MaterialSet  * matlSet,
                           map<string,int> & Qmatls )
{
  //__________________________________
  // Find the material.  Default is matl 0.
  // The user can specify either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  ProblemSpecP module_ps = ps;

  int numMatls = d_materialManager->getNumMatls();

  int defaultMatlIndex = 0;
  
  if( module_ps->findBlock("material") ){
    Material* matl = d_materialManager->parseAndLookupMaterial( module_ps, "material" );
    defaultMatlIndex = matl->getDWIndex();
  } 
  else if ( module_ps->findBlock("materialIndex") ){
    module_ps->get( "materialIndex", defaultMatlIndex );
  } 
  
  vector<int> m;
  m.push_back( defaultMatlIndex );

  //__________________________________
  //  Read in variables label names for optional matl indicies
   ProblemSpecP vars_ps = module_ps->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("PostProcessing: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  for( ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != nullptr; var_spec = var_spec->findNextBlock("analyze") ) {
    map<string,string> attribute;
    var_spec->getAttributes(attribute);

    //__________________________________
    //  Read in the optional material index. It may be different
    //  from the default index
    int matl = defaultMatlIndex;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }

    // bulletproofing
    if(matl < 0 || matl > numMatls){
      throw ProblemSetupException("PostProcessing: problemSetup:: Invalid material index specified for a variable", __FILE__, __LINE__);
    }

    string name = attribute["label"];
    Qmatls[name] = matl;                // matl associated with each Varlabel
    m.push_back(matl);
  }

  //__________________________________
  //  create the matl set
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  matlSet->addAll(m);
  matlSet->addReference();
}


//______________________________________________________________________
//  allocateAndZero
template <class T>
void Module::allocateAndZero( DataWarehouse * new_dw,
                              const VarLabel * label,
                              const int matl,
                              const Patch * patch )
{
  CCVariable<T> Q;
  new_dw->allocateAndPut( Q, label, matl, patch );
  T zero(0.0);
  Q.initialize( zero );
}

//______________________________________________________________________
// Instantiate the explicit template instantiations.
//
template void Module::allocateAndZero<float> ( DataWarehouse  *, const VarLabel *, const int matl, const Patch * );
template void Module::allocateAndZero<double>( DataWarehouse  *, const VarLabel *, const int matl, const Patch * );
template void Module::allocateAndZero<Vector>( DataWarehouse  *, const VarLabel *, const int matl, const Patch * );



//______________________________________________________________________
//
Module::proc0patch0cout::proc0patch0cout( const int nPerTimestep)
{
  d_nTimesPerTimestep = nPerTimestep;
}

//______________________________________________________________________
//
void Module::proc0patch0cout::print(const Patch * patch,
                                   std::ostringstream& msg)
{
  if( d_count <= d_nTimesPerTimestep ){
    if( patch->getID() == 0 && Uintah::Parallel::getMPIRank() == 0 && Uintah::Parallel::getMainThreadID() == std::this_thread::get_id() ){
      std::cout << msg.str();
      d_count += 1;  
    }
  } 
  else {
    d_count = 0;
  }
}


