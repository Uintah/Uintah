/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/MPM/MPMCommon.h> 
#include <Core/Grid/Patch.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Level.h>

using namespace std;
using namespace Uintah;

MPMCommon::MPMCommon(const ProcessorGroup* myworld)
  : d_myworld(myworld)
{
}

MPMCommon::~MPMCommon()
{
}

void MPMCommon::materialProblemSetup(const ProblemSpecP& prob_spec, 
                                     SimulationStateP& sharedState,
                                     MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    string index("");
    ps->getAttribute("index",index);
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;

    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems (redstorm) it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";

    //Create and register as an MPM material
    MPMMaterial *mat = scinew MPMMaterial(ps, sharedState, flags);
    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      sharedState->registerMPMMaterial(mat,index_val);
    }
    else{
      sharedState->registerMPMMaterial(mat);
    }

    // If new particles are to be created, create a copy of each material
    // without the associated geometry
    if (flags->d_createNewParticles) {
      MPMMaterial *mat_copy = scinew MPMMaterial();
      mat_copy->copyWithoutGeom(ps,mat, flags);    
      sharedState->registerMPMMaterial(mat_copy);
    }
  }
}

void MPMCommon::cohesiveZoneProblemSetup(const ProblemSpecP& prob_spec, 
                                         SimulationStateP& sharedState,
                                         MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("cohesive_zone"); ps != 0;
       ps = ps->findNextBlock("cohesive_zone") ) {

    string index("");
    ps->getAttribute("index",index);
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;

    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems (redstorm) it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";

    //Create and register as an MPM material
    CZMaterial *mat = scinew CZMaterial(ps, sharedState, flags);

    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      sharedState->registerCZMaterial(mat,index_val);
    }
    else{
      sharedState->registerCZMaterial(mat);
    }
  }
}
