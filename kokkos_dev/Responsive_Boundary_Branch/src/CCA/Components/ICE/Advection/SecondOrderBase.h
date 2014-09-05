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


#ifndef UINTAH_SECOND_ORDER_BASE_H
#define UINTAH_SECOND_ORDER_BASE_H

#include <CCA/Components/ICE/Advection/Advector.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Ghost.h>

using namespace SCIRun;

#define d_SMALL_NUM 1e-100
//#define DUMP_LIMITER
namespace Uintah {

class SecondOrderBase {

public:
  SecondOrderBase();
  virtual ~SecondOrderBase();
  
  template<class T> 
  void limitedGradient(const CCVariable<T>& q_CC,
                       const Patch* patch,
                       DataWarehouse* new_dw,
                       const CCVariable<vertex<T> >& q_vertex, 
                          CCVariable<T>& q_grad_x,
                          CCVariable<T>& q_grad_y,
                          CCVariable<T>& q_grad_z);
                       
  template<class T> 
  void Q_vertex( const bool useCompatibleFluxes,
                 const CCVariable<T>& q_CC,
                 CCVariable<vertex<T> >& q_vertex,
                 const Patch* patch,
                  const CCVariable<T>& q_grad_x,
                  const CCVariable<T>& q_grad_y,
                  const CCVariable<T>& q_grad_z);
  
                                 
  template<class T> 
    inline void q_CCMaxMin(const CCVariable<T>& q_CC,
                           const Patch* patch,
                              CCVariable<T>& q_CC_max, 
                              CCVariable<T>& q_CC_min);
                           
  template<class T>                                    
  void gradQ( const CCVariable<T>& q_CC,
              const Patch* patch,
               CCVariable<T>& q_grad_x,
               CCVariable<T>& q_grad_y,
               CCVariable<T>& q_grad_z);
 
  void mass_massVertex_ratio( const CCVariable<double>& mass_CC,
                            const Patch* patch,
                               const CCVariable<double>& q_grad_x,
                               const CCVariable<double>& q_grad_y,
                               const CCVariable<double>& q_grad_z);

  template <class T>
  bool 
  flux_to_primitive( const bool useCompatibleFluxes,
                     const bool is_Q_mass_specific,
                     const Patch* patch,
                     const CCVariable<T>& A_CC,
                     const CCVariable<double>& mass,
                     CCVariable<T>& q_CC);
                       
  CCVariable<fflux>  r_out_x, r_out_y, r_out_z;
  CCVariable<vertex<double> > d_mass_massVertex;
  
  bool d_smokeOnOff; 
};

/*_____________________________________________________________________
 Function~ gradQ
 Purpose   Find the x, y, z gradients of q_CC, centered differencing 
_____________________________________________________________________*/
template <class T>
inline void
SecondOrderBase::gradQ( const CCVariable<T>& q_CC,
                        const Patch* patch,
                           CCVariable<T>& q_grad_x,
                           CCVariable<T>& q_grad_y,
                           CCVariable<T>& q_grad_z)
{
  T zero(0.0);
  q_grad_x.initialize(zero);
  q_grad_y.initialize(zero);
  q_grad_z.initialize(zero);

  Vector dx = patch->dCell();
  Vector inv_2del;
  inv_2del.x(1.0/(2.0 * dx.x()) );
  inv_2del.y(1.0/(2.0 * dx.y()) );
  inv_2del.z(1.0/(2.0 * dx.z()) );

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells

  int NGC =1;  // numer of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
    const IntVector& c = *iter;
    
    IntVector r = c + IntVector( 1, 0, 0);
    IntVector l = c + IntVector(-1, 0, 0);
    IntVector t = c + IntVector( 0, 1, 0);
    IntVector b = c + IntVector( 0,-1, 0);
    IntVector f = c + IntVector( 0, 0, 1);    
    IntVector bk= c + IntVector( 0, 0,-1);

    q_grad_x[c] = (q_CC[r] - q_CC[l]) * inv_2del.x();
    q_grad_y[c] = (q_CC[t] - q_CC[b]) * inv_2del.y();
    q_grad_z[c] = (q_CC[f] - q_CC[bk])* inv_2del.z();
  }

  //__________________________________
  // Iterate over the coarsefine interface faces
  // use one-sided first order differencing to compute the gradient
 vector<Patch::FaceType>  faces;
 patch->getCoarseFaces(faces);
 
 vector<Patch::FaceType>::const_iterator f_iter;   
 
  for (f_iter  = faces.begin(); f_iter != faces.end(); ++f_iter){
    Patch::FaceType face = *f_iter;
    IntVector axes = patch->getFaceAxes(face);
    int P_dir = axes[0]; // find the principal dir
    IntVector offset(0,0,0);
    offset[P_dir] = 1;
    
    Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
    CellIterator faceCells =patch->getFaceIterator(face, PEC);
    for (CellIterator itr=faceCells; !itr.done(); itr++) {
      const IntVector& c = *itr;
      q_grad_x[c] = 0.0;
      q_grad_y[c] = 0.0;
      q_grad_z[c] = 0.0;
    }
    
    switch (face) {
    case Patch::xplus:
      for (CellIterator itr=faceCells; !itr.done(); itr++) {
        const IntVector& c = *itr;
        q_grad_x[c] = (q_CC[c] - q_CC[c - offset])/dx.x();
      }
      break;
    case Patch::xminus:
      for (CellIterator itr=faceCells; !itr.done(); itr++) {  
        const IntVector& c = *itr;
        q_grad_x[c] = (q_CC[c + offset] - q_CC[c])/dx.x();
      }
      break;
    case Patch::yplus:
      for (CellIterator itr=faceCells; !itr.done(); itr++) { 
        const IntVector& c = *itr;
        q_grad_y[c] = (q_CC[c] - q_CC[c - offset])/dx.y();
      }
      break;
    case Patch::yminus:
      for (CellIterator itr=faceCells; !itr.done(); itr++) {
        const IntVector& c = *itr;
        q_grad_y[c] = (q_CC[c + offset] - q_CC[c])/dx.y();
      }
      break;
    case Patch::zplus:
      for (CellIterator itr=faceCells; !itr.done(); itr++) {
        const IntVector& c = *itr;
        q_grad_z[c] = (q_CC[c] - q_CC[c - offset])/dx.z();
      }
      break;
    case Patch::zminus:
      for (CellIterator itr=faceCells; !itr.done(); itr++) {
        const IntVector& c = *itr;
        q_grad_z[c] = (q_CC[c + offset] - q_CC[c])/dx.z();
      }
      break;
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break; 
    }
  }
}
/*_____________________________________________________________________
Function~  q_CCMaxMin
Purpose~   This calculates the max and min values of q_CC from the surrounding
           cells
_____________________________________________________________________*/
template <class T>
inline void
SecondOrderBase::q_CCMaxMin(const CCVariable<T>& q_CC,
                            const Patch* patch,
                               CCVariable<T>& q_CC_max, 
                               CCVariable<T>& q_CC_min)
{  
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  

    const IntVector& c = *iter;
       
    IntVector r = c + IntVector( 1, 0, 0);
    IntVector l = c + IntVector(-1, 0, 0);
    IntVector t = c + IntVector( 0, 1, 0);
    IntVector b = c + IntVector( 0,-1, 0);
    IntVector f = c + IntVector( 0, 0, 1);    
    IntVector bk= c + IntVector( 0, 0,-1); 

    T max_tmp, min_tmp;
    max_tmp = Max(q_CC[r], q_CC[l]);
    min_tmp = Min(q_CC[r], q_CC[l]);

    max_tmp = Max(max_tmp, q_CC[t]);
    min_tmp = Min(min_tmp, q_CC[t]);

    max_tmp = Max(max_tmp, q_CC[b]);
    min_tmp = Min(min_tmp, q_CC[b]);

    max_tmp = Max(max_tmp, q_CC[f]);
    min_tmp = Min(min_tmp, q_CC[f]);

    max_tmp = Max(max_tmp, q_CC[bk]);
    min_tmp = Min(min_tmp, q_CC[bk]);

    q_CC_max[c] = max_tmp;
    q_CC_min[c] = min_tmp;
  }

  //__________________________________
  //Coarse fine interface faces
  vector<Patch::FaceType>  faces;
  patch->getCoarseFaces(faces);
  IntVector cl = patch->getExtraCellLowIndex();
  IntVector ch = patch->getExtraCellHighIndex() - IntVector(1,1,1);
  Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
  
  vector<Patch::FaceType>::const_iterator f_iter;   
  for (f_iter  = faces.begin(); f_iter != faces.end(); ++f_iter){
    Patch::FaceType face = *f_iter; 
  
    for (CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
      const IntVector& c = *iter;
      T max_tmp, min_tmp;
      
      max_tmp = T(-DBL_MAX);
      min_tmp = T(DBL_MAX);

      for (int dir=0; dir<3; dir++ ) {
        IntVector R = c;
        IntVector L = c;
        R[dir] += 1;
        L[dir] -= 1;
        
        // intersection with the patch boundary
        L[dir] = Max(cl[dir], L[dir]);
        R[dir] = Min(ch[dir], R[dir]);
         
        max_tmp = Max(max_tmp, q_CC[R]);
        min_tmp = Min(min_tmp, q_CC[L]);
      }
      q_CC_max[c] = max_tmp;
      q_CC_min[c] = min_tmp;
    }
  }  //face loop
}                                                   

/*_____________________________________________________________________
 Function~ q_vertex
 Purpose   compute q_vertex with gradient limited gradients 
           Equation 3.3.5 of reference.  Note that there 2 different
           equations depending on if q_CC is mass specific and we're using
           compatible flux formulation
_____________________________________________________________________*/
template<class T> 
void
SecondOrderBase::Q_vertex( const bool usingCompatibleFluxes,
                           const CCVariable<T>& q_CC,
                           CCVariable<vertex<T> >& q_vertex,
                           const Patch* patch,
                              const CCVariable<T>& q_grad_x,
                              const CCVariable<T>& q_grad_y,
                              const CCVariable<T>& q_grad_z)
{
  Vector dx = patch->dCell();
  double dx_2 = dx.x()/2.0;  // distance from cell center to vertex
  double dy_2 = dx.y()/2.0;
  double dz_2 = dx.z()/2.0;
  int NGC =1;  // number of ghostCells
  
  if(!usingCompatibleFluxes) {        // non-compatible advection
    for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) { 
      const IntVector& c = *iter;

      T xterm = q_grad_x[c] * dx_2;
      T yterm = q_grad_y[c] * dy_2;
      T zterm = q_grad_z[c] * dz_2;
      T q = q_CC[c];
      
      vertex<T>& q_vrtx = q_vertex[c];
      q_vrtx.d_vrtx[0] = q + (-xterm - yterm - zterm);
      q_vrtx.d_vrtx[1] = q + (-xterm - yterm + zterm);
      q_vrtx.d_vrtx[2] = q + (-xterm + yterm - zterm);     
      q_vrtx.d_vrtx[3] = q + (-xterm + yterm + zterm);
      q_vrtx.d_vrtx[4] = q + (xterm - yterm - zterm);
      q_vrtx.d_vrtx[5] = q + (xterm - yterm + zterm);    
      q_vrtx.d_vrtx[6] = q + (xterm + yterm - zterm);
      q_vrtx.d_vrtx[7] = q + (xterm + yterm + zterm);
    }
  }
  if(usingCompatibleFluxes) {         // compatible advection
    for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
      const IntVector& c = *iter;
      T xterm = q_grad_x[c] * dx_2;
      T yterm = q_grad_y[c] * dy_2;
      T zterm = q_grad_z[c] * dz_2;
      
      T q = q_CC[c];
      vertex<T>& q_vrtx = q_vertex[c];
      const vertex<double>& mass_ratio = d_mass_massVertex[c];
      
      q_vrtx.d_vrtx[0] = q + (-xterm - yterm - zterm) * mass_ratio.d_vrtx[0];
      q_vrtx.d_vrtx[1] = q + (-xterm - yterm + zterm) * mass_ratio.d_vrtx[1];
      q_vrtx.d_vrtx[2] = q + (-xterm + yterm - zterm) * mass_ratio.d_vrtx[2];     
      q_vrtx.d_vrtx[3] = q + (-xterm + yterm + zterm) * mass_ratio.d_vrtx[3];
      q_vrtx.d_vrtx[4] = q + (xterm - yterm - zterm)  * mass_ratio.d_vrtx[4];
      q_vrtx.d_vrtx[5] = q + (xterm - yterm + zterm)  * mass_ratio.d_vrtx[5];    
      q_vrtx.d_vrtx[6] = q + (xterm + yterm - zterm)  * mass_ratio.d_vrtx[6];
      q_vrtx.d_vrtx[7] = q + (xterm + yterm + zterm)  * mass_ratio.d_vrtx[7];
    }
  }
}
/*_____________________________________________________________________
 Function~ convertFlux_to_primitive.
 Purpose: - IF using compatible fluxes convert flux into primitive variable  
          - For compatible fluxes, q_CC MUST be a mass-specific quantity
            Test for it.
_____________________________________________________________________*/
template <class T>
bool 
SecondOrderBase::flux_to_primitive( const bool useCompatibleFluxes,
                                    const bool is_Q_mass_specific,
                                    const Patch* patch,
                                    const CCVariable<T>& A_CC,
                                    const CCVariable<double>& mass,
                                    CCVariable<T>& q_CC)
{


  // bulletproofing
  if(useCompatibleFluxes && !is_Q_mass_specific){
    throw InternalError("ICE:SecondOrderAdvection:\n"
    " For compatible fluxes, Q_CC must be a mass-specific quantity \n", __FILE__, __LINE__);
  }
                // compatible fluxes.
  if(useCompatibleFluxes && is_Q_mass_specific) { 
    int NGC =2;  // number of ghostCells
    for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++) {  
     
      const IntVector& c = *iter;
      q_CC[c] = A_CC[c]/mass[c];
    }
    return true;
  } else {      // non-compatible fluxes
    q_CC.copyData(A_CC);
    return false;
  }
}
/*_____________________________________________________________________
 Function~ limitedGradient
_____________________________________________________________________*/
template <class T>
void SecondOrderBase::limitedGradient(const CCVariable<T>& q_CC,
                                      const Patch* patch,
                                      DataWarehouse* new_dw,
                                      const CCVariable<vertex<T> >& q_vertex,
                                          CCVariable<T>& q_grad_x,
                                          CCVariable<T>& q_grad_y,
                                          CCVariable<T>& q_grad_z)
{
  T  frac,temp, zero(0.);
  T  gradLim_max, gradLim_min;
  T unit = T(1.0);
  T SN = T(d_SMALL_NUM);
    
  CCVariable<T> q_CC_max, q_CC_min;
  Ghost::GhostType  gac = Ghost::AroundCells;
  new_dw->allocateTemporary(q_CC_max, patch,gac,1);
  new_dw->allocateTemporary(q_CC_min, patch,gac,1);
   
  q_CCMaxMin( q_CC, patch, q_CC_max, q_CC_min); 

#ifdef DUMP_LIMITER  
  static int counter;
  FILE *fp;
  if(d_smokeOnOff){
    ostringstream fname;
    fname<<"limiter/"<<counter<< ".dat";
    string filename = fname.str();
    
    fp = fopen(filename.c_str(), "w");
    counter +=1;
  }
# endif

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
    const IntVector& c = *iter;

    T q_vrtx_max, q_vrtx_min;
    T Q_CC = q_CC[c];
    
    //__________________________________
    // q_vertex (max/min)   
    q_vrtx_max = q_vertex[c].d_vrtx[0];
    q_vrtx_min = q_vertex[c].d_vrtx[0];
    
    for (int i=1;i<8;i++){
      q_vrtx_max = Max(q_vrtx_max,q_vertex[c].d_vrtx[i]);
      q_vrtx_min = Min(q_vrtx_min,q_vertex[c].d_vrtx[i]);
    }
    
    //__________________________________
    // gradient limiter
    frac = (q_CC_max[c] - Q_CC + SN)/(q_vrtx_max - Q_CC + SN);
    gradLim_max = Max(zero, frac);

    frac = (q_CC_min[c] - Q_CC + SN)/(q_vrtx_min - Q_CC + SN);
    gradLim_min = Max(zero, frac);

    temp = Min(unit, gradLim_max);
    temp = Min(temp, gradLim_min);
    T gradLim = temp;
#ifdef DUMP_LIMITER    
    if(d_smokeOnOff && c.y() == 0 && c.z() == 0){
      fprintf(fp, "%i %16.15E\n",  c.x(),gradLim);
    }
#endif
    
    q_grad_x[c] = q_grad_x[c] * gradLim;
    q_grad_y[c] = q_grad_y[c] * gradLim;
    q_grad_z[c] = q_grad_z[c] * gradLim; 
  }
  
#ifdef DUMP_LIMITER  
  if(d_smokeOnOff){
    fclose(fp);
  }
#endif

}

} // end namespace Uintah

#endif
