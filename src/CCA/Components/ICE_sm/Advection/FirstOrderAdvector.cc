/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <CCA/Components/ICE_sm/Advection/FirstOrderAdvector.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Patch.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Endian.h>
#include <iostream>

//#define SPEW
#undef SPEW
#define is_rightFace_variable(face,var) ( ((face == "xminus" || face == "xplus") && var == "scalar-f") ?1:0  )

using namespace Uintah;
using namespace std;

/* ---------------------------------------------------------------------
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
 ---------------------------------------------------------------------  */
FirstOrderAdvector::FirstOrderAdvector() 
{
}


FirstOrderAdvector::FirstOrderAdvector(DataWarehouse* new_dw, 
                                       const Patch* patch,
                                       const bool isNewGrid)
{
}


FirstOrderAdvector::~FirstOrderAdvector()
{
}

FirstOrderAdvector* FirstOrderAdvector::clone(DataWarehouse* new_dw,
                                              const Patch* patch,
                                              const bool isNewGrid)
{
  return scinew FirstOrderAdvector(new_dw,patch,isNewGrid);
}

/* ---------------------------------------------------------------------
 Function~  influxOutfluxVolume--
 Purpose~   calculate the individual outfluxes for each cell.
            This includes the slabs
 Steps for each cell:  
 1) calculate the volume for each outflux
 2) test to see if the total outflux > cell volume

Implementation notes:
The outflux of volume is calculated in each cell in the computational domain
+ one layer of extra cells  surrounding the domain.The face-centered velocity 
needs to be defined on all faces for these cells 

See schematic diagram at bottom of ice.cc for del* definitions
 ---------------------------------------------------------------------  */

void FirstOrderAdvector::inFluxOutFluxVolume(
                        const SFCXVariable<double>& uvel_FC,
                        const SFCYVariable<double>& vvel_FC,
                        const SFCZVariable<double>& wvel_FC,
                        const double& delT, 
                        const Patch* patch,
                        const int& indx,
                        const bool& bulletProof_test,
                        DataWarehouse* new_dw,
                        advectVarBasket* VB)

{
  Vector dx = patch->dCell();
  double vol = dx.x()*dx.y()*dx.z();
  double delY_top, delY_bottom,delX_right, delX_left, delZ_front, delZ_back;
  double delX = dx.x(), delY = dx.y(), delZ = dx.z();
  
  new_dw->allocateTemporary(VB->OFS,patch, Ghost::AroundCells,1);
  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells  
  bool error = false;
  
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++) {  
    const IntVector& c = *iter;
    
    delY_top    = std::max(0.0, (vvel_FC[c+IntVector(0,1,0)] * delT));
    delY_bottom = std::max(0.0,-(vvel_FC[c                 ] * delT));
    delX_right  = std::max(0.0, (uvel_FC[c+IntVector(1,0,0)] * delT));
    delX_left   = std::max(0.0,-(uvel_FC[c                 ] * delT));
    delZ_front  = std::max(0.0, (wvel_FC[c+IntVector(0,0,1)] * delT));
    delZ_back   = std::max(0.0,-(wvel_FC[c                 ] * delT));
    
    //__________________________________
    //   SLAB outfluxes
    double delX_Z = delX * delZ;
    double delX_Y = delX * delY;
    double delY_Z = delY * delZ;
    fflux& ofs = VB->OFS[c];
    
    ofs.d_fflux[TOP]   = delY_top   * delX_Z;
    ofs.d_fflux[BOTTOM]= delY_bottom* delX_Z;
    ofs.d_fflux[RIGHT] = delX_right * delY_Z;
    ofs.d_fflux[LEFT]  = delX_left  * delY_Z;
    ofs.d_fflux[FRONT] = delZ_front * delX_Y;
    ofs.d_fflux[BACK]  = delZ_back  * delX_Y; 

    //__________________________________
    //  Bullet proofing
    double total_fluxout = 0.0;
    for(int face = TOP; face <= BACK; face++ )  {
      total_fluxout  += ofs.d_fflux[face];
    }
    if(total_fluxout > vol){
      error = true;
    }
  }  //cell iterator
  
  //__________________________________
  // if total_fluxout > vol then 
  // -find the cell, 
  // -set the outflux slab vol in all cells = 0.0,
  // -request that the timestep be restarted.
  // -ignore if a timestep restart has already been requested
  bool tsr = new_dw->timestepRestarted(); 
    
  if (error && bulletProof_test && !tsr) {
    vector<IntVector> badCells;
    vector<fflux>  badOutflux;
    
    for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++) {
      IntVector c = *iter; 
      double total_fluxout = 0.0;
      fflux& ofs = VB->OFS[c];
      
      for(int face = TOP; face <= BACK; face++ )  {
        total_fluxout  += VB->OFS[c].d_fflux[face];
        VB->OFS[c].d_fflux[face] = 0.0;
      }
      // keep track of which cells are bad
      if (vol - total_fluxout < 0.0) {
        badCells.push_back(c);
        badOutflux.push_back(ofs);
      }
    }  // cell iter
    warning_restartTimestep( badCells,badOutflux, vol, indx, patch, new_dw);
  }  // if total_fluxout > vol
  
  if (error && !bulletProof_test) {
    std::ostringstream mesg;
    std::cout << " WARNING: ICE Advection operator Influx/Outflux volume error:"
              << " Patch " << patch->getID()
              << ", Level " << patch->getLevel()->getIndex()<< std::endl;
  }
}

/*_____________________________________________________________________
 Function~ advectQ
_____________________________________________________________________*/
//     M A S S
void FirstOrderAdvector::advectMass(const CCVariable<double>& q_CC,
                                    CCVariable<double>& q_advected,
                                    advectVarBasket* varBasket)
{
        
  advectSlabs<double>(q_CC,varBasket->patch,q_advected, varBasket);       
}

//__________________________________
//     D O U B L E 
// (int_eng, sp_vol * mass, transported Variables)
void FirstOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 CCVariable<double>& q_advected,
                                 advectVarBasket* varBasket)
{                                 
  advectSlabs<double>(q_CC,varBasket->patch,q_advected, varBasket);
}

//__________________________________
//  S P E C I A L I Z E D   D O U B L E 
//  needed by implicit ICE  q_CC = volFrac
void FirstOrderAdvector::advectQ(const CCVariable<double>& q_CC,
                                 const Patch* patch,
                                 CCVariable<double>& q_advected,
                                 advectVarBasket* varBasket,
                                 DataWarehouse* /*new_dw*/)
{
  advectSlabs<double>(q_CC,patch,q_advected,  varBasket);
}
//__________________________________
//     V E C T O R  (momentum)
void FirstOrderAdvector::advectQ(const CCVariable<Vector>& q_CC,
                                 const CCVariable<double>& /*mass*/,
                                 CCVariable<Vector>& q_advected,
                                 advectVarBasket* varBasket)
{
  advectSlabs<Vector>(q_CC,varBasket->patch,q_advected, varBasket);
                      
} 

/*_____________________________________________________________________
 Function~  Advect--  driver program that does the advection  
_____________________________________________________________________*/
template <class T> 
  void FirstOrderAdvector::advectSlabs(const CCVariable<T>& q_CC,
                                       const Patch* patch,                   
                                       CCVariable<T>& q_advected,
                                       advectVarBasket* VB) // function is passed in
{                  
  Vector dx = patch->dCell();            
  double invvol = 1.0/(dx.x() * dx.y() * dx.z());                     

  CCVariable<fflux>& OFS = VB->OFS;  // for brevity
    
  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
    const IntVector& c = *iter;  
    
    T q_face_flux[6];
    double faceVol[6];
    
    T sum_q_face_flux(0.0);   
    for(int f = TOP; f <= BACK; f++ )  {    
      //__________________________________
      //   S L A B S
      // q_CC: vol_frac, mass, momentum, int_eng....
      //      for consistent units you need to divide by cell volume
      // 
      IntVector ac = c + S_ac[f];     // slab adjacent cell
      double outfluxVol = OFS[c ].d_fflux[OF_slab[f]];
      double influxVol  = OFS[ac].d_fflux[IF_slab[f]];

      T q_faceFlux_tmp  =   q_CC[ac] * influxVol - q_CC[c] * outfluxVol;
        
      faceVol[f]       =  outfluxVol +  influxVol;
      q_face_flux[f]   = q_faceFlux_tmp; 
      sum_q_face_flux += q_faceFlux_tmp;
    }  
    q_advected[c] = sum_q_face_flux*invvol;
  }
}
/*_____________________________________________________________________
 Function~ q_FC_operator
 Compute q at the face center.
_____________________________________________________________________*/
template<class T>
void FirstOrderAdvector::q_FC_operator(CellIterator iter, 
                                       IntVector adj_offset,
                                       const int face,
                                       const CCVariable< fflux >& OFS,
                                       const CCVariable<double>& q_CC,
                                       T& q_FC)
{
  for(;!iter.done(); iter++){
    IntVector R = *iter;      
    IntVector L = R + adj_offset; 

     // face:           LEFT,   BOTTOM,   BACK  
     // IF_slab[face]:  RIGHT,  TOP,      FRONT
    double outfluxVol = OFS[R].d_fflux[face];
    double influxVol  = OFS[L].d_fflux[IF_slab[face]];

    double q_faceFlux = q_CC[L] * influxVol - q_CC[R] * outfluxVol;
    double faceVol = outfluxVol + influxVol;

    double q_tmp_FC = fabs(q_faceFlux)/(faceVol + 1.0e-100);

    // if q_tmp_FC = 0.0 then set it equal to q_CC[c]
    q_FC[R] = equalZero(q_faceFlux, q_CC[R], q_tmp_FC);
  }
}

/*_____________________________________________________________________
 Function~  q_FC_PlusFaces
 Compute q_FC values on the faces between the extra cells
 and the interior domain only on the x+, y+, z+ patch faces 
_____________________________________________________________________*/
void FirstOrderAdvector::q_FC_PlusFaces( const CCVariable<double>& q_CC,
                                         const Patch* patch,
                                         SFCXVariable<double>& q_XFC,
                                         SFCYVariable<double>& q_YFC,
                                         SFCZVariable<double>& q_ZFC,
                                         advectVarBasket* vb)
{                                                  
  vector<IntVector> adj_offset(3);
  adj_offset[0] = IntVector(-1, 0, 0);    // X faces
  adj_offset[1] = IntVector(0, -1, 0);    // Y faces
  adj_offset[2] = IntVector(0,  0, -1);   // Z faces
  
  Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
  
  IntVector patchOnBoundary = patch->noNeighborsHigh();
  // only work on patches that are at the edge of the computational domain
  
  if (patchOnBoundary.x() == 1 ){
    CellIterator Xiter=patch->getFaceIterator(Patch::xplus,MEC);
    q_FC_operator<SFCXVariable<double> >(Xiter, adj_offset[0], LEFT,  
                                         vb->OFS, q_CC,q_XFC);
  } 
  if (patchOnBoundary.y() == 1 ){
    CellIterator Yiter=patch->getFaceIterator(Patch::yplus,MEC);
    q_FC_operator<SFCYVariable<double> >(Yiter, adj_offset[1], BOTTOM,
                                         vb->OFS, q_CC,q_YFC); 
  }
  if (patchOnBoundary.z() == 1 ){  
    CellIterator Ziter=patch->getFaceIterator(Patch::zplus,MEC);
    q_FC_operator<SFCZVariable<double> >(Ziter, adj_offset[2], BACK,  
                                         vb->OFS, q_CC,q_ZFC);  
  }
}
/*_____________________________________________________________________
 Function~ q_FC_flux_operator
 Compute the flux of q across a face.  The flux is need by the AMR 
 refluxing operation
_____________________________________________________________________*/
template<class T, class V>
void FirstOrderAdvector::q_FC_flux_operator(CellIterator iter, 
                                          IntVector adj_offset,
                                          const int face,
                                          const CCVariable< fflux >& OFS,
                                          const CCVariable< V >& q_CC,
                                          T& q_FC_flux )
{
  int out_indx = OF_slab[face]; //LEFT,   BOTTOM,   BACK 
  int in_indx  = IF_slab[face]; //RIGHT,  TOP,      FRONT

  for(;!iter.done(); iter++){
    IntVector c = *iter;      
    IntVector ac = c + adj_offset; 

    double outfluxVol = OFS[c].d_fflux[out_indx];
    double influxVol  = OFS[ac].d_fflux[in_indx];

    q_FC_flux[c] += q_CC[ac] * influxVol - q_CC[c] * outfluxVol;
    
  }    
}
