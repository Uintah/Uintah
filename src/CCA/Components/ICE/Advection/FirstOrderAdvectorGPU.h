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


#ifndef UINTAH_FIRST_ORDER_ADVECTOR_GPU_H
#define UINTAH_FIRST_ORDER_ADVECTOR_GPU_H

#include <CCA/Components/ICE/Advection/Advector.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Disclosure/TypeDescription.h>

#include <sci_defs/cuda_defs.h>

namespace Uintah {


  class FirstOrderAdvectorGPU : public Advector {

  public:
    FirstOrderAdvectorGPU();
    FirstOrderAdvectorGPU(DataWarehouse* new_dw, 
                       const Patch* patch,
                       const bool isNewGrid);
    virtual ~FirstOrderAdvectorGPU();

    virtual FirstOrderAdvectorGPU* clone(DataWarehouse* new_dw, 
                                      const Patch* patch,
                                      const bool isNewGrid);


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_CC,
                                     const SFCYVariable<double>& vvel_CC,
                                     const SFCZVariable<double>& wvel_CC,
                                     const double& delT, 
                                     const Patch* patch,
                                     const int&  indx,
                                     const bool& bulletProof_test,
                                     DataWarehouse* new_dw);

    virtual void inFluxOutFluxVolumeGPU(const SFCXVariable<double>& uvel_FC,
                                        const SFCYVariable<double>& vvel_FC,
                                        const SFCZVariable<double>& wvel_FC,
                                        const double& delT,
                                        const Patch* patch,
                                        const int& indx,
                                        const bool& bulletProofing_test,
                                        DataWarehouse* new_dw,
                                        const int& device);

    virtual void  advectQ(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          CCVariable<double>& q_advected,
                          advectVarBasket* varBasket,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC,
                          DataWarehouse* /*new_dw*/);
 
    virtual void advectQ(const CCVariable<double>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<double>& q_advected,
                         advectVarBasket* vb);
 
    virtual void advectQ(const CCVariable<Vector>& q_CC,
                         const CCVariable<double>& mass,
                         CCVariable<Vector>& q_advected,
                         advectVarBasket* vb); 
                         
    virtual void advectMass(const CCVariable<double>& mass,
                            CCVariable<double>& q_advected,
                            advectVarBasket* vb);
    
  private:
    CCVariable<fflux> d_OFS;
    
    friend class FirstOrderCEAdvector;

  private:
 
    template<class T, typename F> 
      void advectSlabs(const CCVariable<T>& q_CC,
                       const Patch* patch,
                       CCVariable<T>& q_advected,
                       SFCXVariable<double>& q_XFC,
                       SFCYVariable<double>& q_YFC,
                       SFCZVariable<double>& q_ZFC,
                       F function); 
                                                       
    template<class T>
      void q_FC_operator(CellIterator iter, 
                         IntVector adj_offset,
                         const int face,
                         const CCVariable<double>& q_CC,
                         T& q_FC);
                        
      void q_FC_PlusFaces(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC);
                                  
    template<class T, class V>
      void q_FC_flux_operator(CellIterator iter, 
                              IntVector adj_offset,
                              const int face,
                              const CCVariable<V>& q_CC,
                              T& q_FC);
                                               
    template<class T>
      void q_FC_fluxes(const CCVariable<T>& q_CC, 
                       const string& desc,
                       advectVarBasket* vb);
  };

/*
      
    ///////////////////////////
    // GPU Kernel Prototypes //
    ///////////////////////////
    /// @brief The kernel that computes influx and outflux values essentially replacing the cell iterator.
    /// @param domainSize A three component vector that gives the size of the domain as (x,y,z)
    /// @param domainLower A three component vector that gives the lower corner of the work area as (x,y,z)
    /// @param cellSizes A three component vector containing dx, dy and dz for the cell
    /// @param ghostLayers The number of layers of ghost cells
    /// @param delt The change in time from the last timestep
    /// @param uvel_FC The pointer to face velocities in x
    /// @param vvel_FC The pointer to face velocities in y
    /// @param wvel_FC The pointer to face velocities in z
    /// @param OFS Pointer to a six component array that must be updated in this function.
    __global__ void inFluxOutFluxVolumeKernel(uint3 domainSize,
                                              uint3 domainLower,
                                              uint3 cellSizes,
                                              int ghostLayers,
                                              double delt,
                                              double *uvel_FC, 
                                              double *vvel_FC, 
                                              double *wvel_FC,
                                              double **OFS);
      
    /// @brief A kernel that applies the advection operation to a number of slabs.
    /// @param domainSize A three component vector that gives the size of the domain as (x,y,z)
    /// @param domainLower A three component vector that gives the lower corner of the work area as (x,y,z)
    /// @param ghostLayers The number of layers of ghost cells
    /// @param q_CC Pointer to the advected variable
    /// @param q_advected Pointer to the amount of q_CC advected
    /// @param q_XFC Pointer to the advection through x faces
    /// @param q_YFC Pointer to the advection through y faces
    /// @param q_ZFC Pointer to the advection through z faces
    /// @param OFS Pointer to a six component array that must be updated in this function.
    /// @param invol Inverse of the volume of a single cell
    __global__ void advectSlabsKernel(uint3 domainSize,
                                      uint3 domainLower,
                                      int ghostLayers,
                                      double *q_CC,
                                      double *q_advected,
                                      double *q_XFC,
                                      double *q_YFC,
                                      double *q_ZFC,
                                      double **OFS,
                                      double invol);
      
    /// @brief A kernel that computes the total flux through a face. 
    /// @param domainSize A three component vector that gives the size of the domain as (x,y,z)
    /// @param domainLower A three component vector that gives the lower corner of the work area as (x,y,z)
    /// @param adjOffset Offset is x, y or z
    /// @param ghostLayers The number of layers of ghost cells
    /// @param face the current face being investigated
    /// @param oppositeFace the face opposite "face"
    /// @param OFS a pointer to the six component array that is used for fluxes
    /// @param q_CC Pointer to q_CC allocated on the device
    /// @param q_FC Pointer to q_[X,Y,Z]FC allocated on the device
    __global__ void q_FC_operatorKernel(uint3 domainSize,
                                        uint3 domainLower,
                                        uint3 adjOffset,
                                        int ghostLayers,
                                        int face,
                                        int oppositeFace,
                                        double **OFS,
                                        double *q_CC,
                                        double *q_FC);
    
      /// @brief A kernel that computes the flux of q across a face.  The flux is need by the AMR refluxing operation.
      /// @param domainSize A three component vector that gives the size of the domain as (x,y,z)
      /// @param domainLower A three component vector that gives the lower corner of the work area as (x,y,z)
      /// @param adjOffset Offset is x, y or z
      /// @param ghostLayers The number of layers of ghost cells
      /// @param q_CC Pointer to q_CC allocated on the device
      /// @param q_FC Pointer to q_[X,Y,Z]FC allocated on the device
      __global__ void q_FC_flux_operatorKernel(uint3 domainSize,
                                               uint3 domainLower,
                                               uint3 adjOffset,
                                               int ghostLayers,
                                               int face,
                                               int oppositeFace,
                                               double **OFS,
                                               double *q_CC,
                                               double *q_FC_flux);
*/
}

#endif


