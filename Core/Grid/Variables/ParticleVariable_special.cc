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


#ifndef Uintah_Core_Grid_ParticleVariable_special_cc
#define Uintah_Core_Grid_ParticleVariable_special_cc

#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;


  template<>
  void
  ParticleVariable<Point>::packMPI(void* buf, int bufsize, int* bufpos,
                                   const ProcessorGroup* pg,
                                   ParticleSubset* pset, const Patch* forPatch)
  {
    // This should be fixed for variable sized types!
    if (!forPatch->isVirtual()) {
      packMPI(buf, bufsize, bufpos, pg, pset);
    }
    else {
      Vector offset = forPatch->getVirtualOffsetVector();
      const TypeDescription* td = getTypeDescription()->getSubType();
      if(td->isFlat()){
        for(ParticleSubset::iterator iter = pset->begin();
            iter != pset->end(); iter++){
          Point p = d_pdata->data[*iter] - offset;
          MPI_Pack(&p, 1, td->getMPIType(), buf, bufsize, bufpos, pg->getComm());
        }
      } else {
        SCI_THROW(InternalError("packMPI not finished\n", __FILE__, __LINE__));
      }
    }
  }

  // specialization for T=Point
  template <>
  void ParticleVariable<Point>::gather(ParticleSubset* pset,
                                       const vector<ParticleSubset*> &subsets,
                                       const vector<ParticleVariableBase*> &srcs,
                                       const vector<const Patch*>& srcPatches,
                                       particleIndex extra)
  {
    
    if(d_pdata && d_pdata->removeReference())
      delete d_pdata;
    if(d_pset && d_pset->removeReference())
      delete d_pset;

    // pset center patch and neighbor patch are not in same level
    // (probably on an AMR copy data timestep)
    const Patch* patch = pset->getPatch();
    if (pset->getNeighbors().size() != 0 && 
        pset->getNeighbors().front()->getLevel() != patch->getLevel()){
      patch = srcPatches[0];
    }
    
    IntVector lowIndex(pset->getLow()), highIndex(pset->getHigh());
    
    d_pset = pset;
    pset->addReference();
    
    d_pdata = scinew ParticleData<Point>(pset->numParticles());
    d_pdata->addReference();
    
    ASSERTEQ(subsets.size(), srcs.size());
    ParticleSubset::iterator dst_iter = pset->begin();
    
    for(int i=0;i<(int)subsets.size();i++){
      ParticleVariable<Point>* srcptr = dynamic_cast<ParticleVariable<Point>*>(srcs[i]);
      
      if(!srcptr){
        SCI_THROW(TypeMismatchException("Type mismatch in ParticleVariable::gather", __FILE__, __LINE__));
      }
      
      ParticleVariable<Point>& src = *srcptr;
      ParticleSubset* subset = subsets[i];
      const Patch* srcPatch  = srcPatches[i];
      
      // no or real srcPatch
      if (srcPatch == 0 || !srcPatch->isVirtual()) {
        for(ParticleSubset::iterator src_iter = subset->begin(); src_iter != subset->end(); src_iter++){
        
          (*this)[*dst_iter] = src[*src_iter];         
      
          IntVector ptIndex =  patch->getLevel()->getCellIndex(src[*src_iter]);

          ASSERT( Patch::containsIndex(lowIndex,highIndex,ptIndex));
          dst_iter++;
        }
      }
      else if (subset->numParticles() != 0) {
        Vector offset = srcPatch->getVirtualOffsetVector();
      
        for(ParticleSubset::iterator src_iter = subset->begin();
            src_iter != subset->end(); src_iter++){
            
          (*this)[*dst_iter] = src[*src_iter] + offset;
          
          
          IntVector ptIndex =  patch->getLevel()->getCellIndex((*this)[*dst_iter]);
          ASSERT( Patch::containsIndex(lowIndex,highIndex,ptIndex));
          dst_iter++;
        }
      }
    }  // subset loop
    ASSERTEQ(dst_iter+extra,pset->end());    
  }


  template<>
  void
  ParticleVariable<double>::emitNormal(ostream& out, const IntVector&,
                                  const IntVector&, ProblemSpecP varnode, bool outputDoubleAsFloat )
  {
    const TypeDescription* td = fun_getTypeDescription((double*)0);

    if (varnode->findBlock("numParticles") == 0) {
      varnode->appendElement("numParticles", d_pset->numParticles());
    }
    if(!td->isFlat()){
      SCI_THROW(InternalError("Cannot yet write non-flat objects!\n", __FILE__, __LINE__));
    }
    else {
      if (outputDoubleAsFloat) {
        // This could be optimized...
        ParticleSubset::iterator iter = d_pset->begin();
        for ( ; iter != d_pset->end(); iter++) {
          float tempFloat = (float)(*this)[*iter];
          out.write((char*)&tempFloat, sizeof(float));
        }
      } else {
        // This could be optimized...
        ParticleSubset::iterator iter = d_pset->begin();
        while(iter != d_pset->end()){
          particleIndex start = *iter;
          iter++;
          particleIndex end = start+1;
          while(iter != d_pset->end() && *iter == end) {
            end++;
            iter++;
          }
          ssize_t size = (ssize_t)(sizeof(double)*(end-start));
          out.write((char*)&(*this)[start], size);
        }
      }
    }
  }


#endif // this file does need to be included to satisfy template instantiations
       // for some compilers
