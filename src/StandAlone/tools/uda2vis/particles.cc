/*

   The MIT License

   Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#include <stdio.h>

#include <StandAlone/tools/uda2vis/particles.h>

#include <Core/Geometry/Point.h>

using namespace SCIRun;


//////////////////////////////////////////////////////////////////////////////////

template<>
void
handleParticleData<Point>(QueryInfo & qinfo, int matlNo,
                          ParticleVariableRaw &result,
                          string varSelected, int patchNo)
{
  const Patch* patch = qinfo.level->getPatch(patchNo);
  ConsecutiveRangeSet matlsForVar;
  if (matlNo<0)
    matlsForVar = qinfo.archive->queryMaterials(varSelected, patch, qinfo.timestep);
  else
    matlsForVar.addInOrder(matlNo);

  for( ConsecutiveRangeSet::iterator matlIter = matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ ) {
    int matl = *matlIter;

    ParticleVariable<Point> value;
    qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );

    ParticleSubset* pset = value.getParticleSubset();
    if (!pset) { 
      printf("not sure if this case is handled correctly....\n");
      exit( 1 );
    }

    int numParticles = pset->numParticles();
    if (numParticles > 0) {
      for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
        result.values.push_back( (float)value[*iter].x() );
        result.values.push_back( (float)value[*iter].y() );
        result.values.push_back( (float)value[*iter].z() );
      }
    }

    patchMatlPart pObj(patch->getID(), matl, numParticles);
  }

  result.components = 3;
}


template<>
void
handleParticleData<Vector>(QueryInfo & qinfo, int matlNo,
                           ParticleVariableRaw &result,
                           string varSelected, int patchNo )
{
  const Patch* patch = qinfo.level->getPatch(patchNo);
  ConsecutiveRangeSet matlsForVar;
  if (matlNo<0)
    matlsForVar = qinfo.archive->queryMaterials(varSelected, patch, qinfo.timestep);
  else
    matlsForVar.addInOrder(matlNo);

  for( ConsecutiveRangeSet::iterator matlIter = matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ ) {
    int matl = *matlIter;

    ParticleVariable<Vector> value;
    qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );

    ParticleSubset* pset = value.getParticleSubset();
    if (!pset) {
      printf("NOT sure that this case is being handled correctly...\n");
      exit( 1 );
    }

    int numParticles = pset->numParticles();
    if (numParticles > 0) {
      for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
        result.values.push_back( (float)value[*iter].x() );
        result.values.push_back( (float)value[*iter].y() );
        result.values.push_back( (float)value[*iter].z() );
      }
    }
  }

  result.components = 3;
}



template<>
void
handleParticleData<Matrix3>(QueryInfo & qinfo, int matlNo,
                            ParticleVariableRaw &result,
                            string varSelected, int patchNo)
{
  const Patch* patch = qinfo.level->getPatch(patchNo);
  ConsecutiveRangeSet matlsForVar;
  if (matlNo<0)
    matlsForVar = qinfo.archive->queryMaterials(varSelected, patch, qinfo.timestep);
  else
    matlsForVar.addInOrder(matlNo);

  for( ConsecutiveRangeSet::iterator matlIter = matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ ) {
    int matl = *matlIter;

    ParticleVariable<Matrix3> value;
    qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );

    ParticleSubset* pset = value.getParticleSubset();
    if (!pset) { 
      printf("not sure if this case is handled correctly....\n");
      exit( 1 );
    }
      
    int numParticles = pset->numParticles();
    if (numParticles > 0) {
      for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
        for (unsigned int i = 0; i < 3; i++) {
          for (unsigned int j = 0; j < 3; j++) {
            result.values.push_back(value[*iter](i,j));
          }
        }
      }
    }
  }

  result.components=9;
}


template<class PartT>
void
handleParticleData(QueryInfo & qinfo, int matlNo,
                   ParticleVariableRaw &result,
                   string varSelected, int patchNo )
{
  const Patch* patch = qinfo.level->getPatch(patchNo);
  ConsecutiveRangeSet matlsForVar;
  if (matlNo<0)
    matlsForVar = qinfo.archive->queryMaterials(varSelected, patch, qinfo.timestep);
  else
    matlsForVar.addInOrder(matlNo);

  for( ConsecutiveRangeSet::iterator matlIter = matlsForVar.begin(); matlIter != matlsForVar.end(); matlIter++ ) {
    int matl = *matlIter;

    ParticleVariable<PartT> value;
    qinfo.archive->query( value, qinfo.varname, matl, patch, qinfo.timestep );

    ParticleSubset* pset = value.getParticleSubset();
    if (!pset) {
      printf("NOT sure that this case is being handled correctly...\n");
      exit( 1 );
    }

    int numParticles = pset->numParticles();
    if (numParticles > 0) {
      for( ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
        result.values.push_back( (float) value[*iter] );
      }
    } 
  }

  result.components = 1;
}





///////////////////////////////////////////////////////////////////////////////
// Instantiate some of the needed verisons of functions.

template void handleParticleData<int>    (QueryInfo&, int, ParticleVariableRaw&, string, int);
template void handleParticleData<long64> (QueryInfo&, int, ParticleVariableRaw&, string, int);
template void handleParticleData<float>  (QueryInfo&, int, ParticleVariableRaw&, string, int);
template void handleParticleData<double> (QueryInfo&, int, ParticleVariableRaw&, string, int);
template void handleParticleData<Point>  (QueryInfo&, int, ParticleVariableRaw&, string, int);
template void handleParticleData<Vector> (QueryInfo&, int, ParticleVariableRaw&, string, int);
template void handleParticleData<Matrix3>(QueryInfo&, int, ParticleVariableRaw&, string, int);
