#include <Packages/Uintah/CCA/Components/MPM/MPMCommon.h> 
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Level.h>


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
  ProblemSpecP mat_ps =  prob_spec->findBlock("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0;
       ps = ps->findNextBlock("material") ) {
    string index("");
    ps->getAttribute("index",index);
    stringstream id(index);
    int index_val = -1;
    id >> index_val;
    //cout << "Material attribute = " << index_val << endl;

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


void MPMCommon::printSchedule(const PatchSet* patches,
                              DebugStream& dbg,
                              const string& where)
{
  if (dbg.active()){
    dbg << d_myworld->myrank() << " " 
        << where << "L-"
        << getLevel(patches)->getIndex()<< endl;
  }  
}

void MPMCommon::printSchedule(const LevelP& level,
                              DebugStream& dbg,
                              const string& where)
{
  if (dbg.active()){
    dbg << d_myworld->myrank() << " " 
        << where << "L-"
        << level->getIndex()<< endl;
  }  
}

void MPMCommon::printTask(const PatchSubset* patches,
                          const Patch* patch,
                          DebugStream& dbg,
                          const string& where)
{
  if (dbg.active()){
    dbg << d_myworld->myrank() << " " 
        << where << " MPM \tL-"
        << getLevel(patches)->getIndex()
        << " patch " << patch->getGridIndex()<< endl;
  }  
}
