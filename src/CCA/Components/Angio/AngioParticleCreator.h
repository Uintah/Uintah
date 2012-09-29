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

#ifndef __ANGIO_PARTICLE_CREATOR_H__
#define __ANGIO_PARTICLE_CREATOR_H__

#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <vector>
#include <map>

namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;

  class GeometryObject;
  class Patch;
  class DataWarehouse;
  class AngioLabel;
  class ParticleSubset;
  class VarLabel;
  class AngioMaterial;
  class AngioFlags;

  class AngioParticleCreator {
  public:
    
    AngioParticleCreator(AngioMaterial* matl, AngioFlags* flags);

    virtual ~AngioParticleCreator();

    virtual ParticleSubset* createParticles(AngioMaterial* matl,
                                            std::string i_frag_file,
                                            particleIndex numParticles,
                                            CCVariable<short int>& cellNAPID,
                                            const Patch*,DataWarehouse* new_dw);

    virtual ParticleSubset* allocateVariables(particleIndex numParticles,
                                              int dwi, const Patch* patch,
                                              DataWarehouse* new_dw);

//    virtual void allocateVariablesAddRequires(Task* task,
//                                              const MPMMaterial* matl,
//                                              const PatchSet* patch) const;

     void allocateVariablesAdd(DataWarehouse* new_dw,
                               ParticleSubset* addset,
                               map<const VarLabel*,
                               ParticleVariableBase*>* newState,
                               vector<Point>& x_new, vector<Vector>& growth_new,

                               vector<double>& l_new,   vector<double>& rad_new,
                               vector<double>& ang_new, vector<double>& t_new,
                               vector<double>& r_b_new,
                               vector<double>& pmass_new,
                               vector<double>& pvol_new,vector<int>& pt0_new,
                               vector<int>& pt1_new,    vector<int>& par_new,
                               vector<IntVector>& vcell_idx,
                               CCVariable<short int>& cellNAPID);

    virtual void registerPermanentParticleState();

    virtual particleIndex countParticles(const Patch*, std::string i_frag_file);

    vector<const VarLabel* > returnParticleState();
    vector<const VarLabel* > returnParticleStatePreReloc();

  protected:

    virtual void initializeParticle(const Patch* patch,
                                    AngioMaterial* matl,
                                    Point p1, Point p2, IntVector cell_idx,
                                    particleIndex i,
                                    CCVariable<short int>& cellNAPI);
    
    ParticleVariable<Point>  position0;
    ParticleVariable<Vector> growth;
    ParticleVariable<double> pmass,pvolume,radius,phi,length,tofb,recentbranch;
    ParticleVariable<long64> pparticleID;
    ParticleVariable<int>    tip0,tip1,parent;

    AngioLabel* d_lb;

    vector<const VarLabel* > particle_state, particle_state_preReloc;
    typedef map<pair<const Patch*,GeometryObject*>,vector<Point> > geompoints;
    geompoints d_object_points;
  };

} // End of namespace Uintah

#endif // __PARTICLE_CREATOR_H__
