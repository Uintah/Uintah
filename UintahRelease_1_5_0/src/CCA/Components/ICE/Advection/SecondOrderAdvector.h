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

#ifndef UINTAH_SECOND_ORDER_ADVECTOR_H
#define UINTAH_SECOND_ORDER_ADVECTOR_H

#include <CCA/Components/ICE/Advection/Advector.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/ICE/Advection/SecondOrderBase.h>
#include <Core/Disclosure/TypeDescription.h>

#include <Core/Containers/StaticArray.h>

namespace Uintah {

  class SecondOrderAdvector : public Advector, public SecondOrderBase {

  public:
    SecondOrderAdvector();
    SecondOrderAdvector(DataWarehouse* new_dw, 
                        const Patch* patch,
                        const bool isNewGrid);
                        
    virtual ~SecondOrderAdvector();

    virtual SecondOrderAdvector* clone(DataWarehouse* new_dw, 
                                       const Patch* patch,
                                       const bool isNewGrid);


    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_CC,
                                     const SFCYVariable<double>& vvel_CC,
                                     const SFCZVariable<double>& wvel_CC,
                                     const double& delT, 
                                     const Patch* patch,
                                     const int& indx,
                                     const bool& bulletProof_test,
                                     DataWarehouse* new_dw);
                         
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
    CCVariable<fflux> d_OFS;  // outflux slabs
    CCVariable<facedata<double> > d_mass_slabs;  // outflux slabs

  private:                                 
    template<class T> 
      void qAverageFlux(const bool useCompatibleFluxes,
                        const CCVariable<T>& q_CC,
                        const CCVariable<double>& mass_CC,
                        const Patch* patch,
                        CCVariable<facedata<T> >& q_OAFS,
                        const CCVariable<T>& grad_x,
                        const CCVariable<T>& grad_y,
                        const CCVariable<T>& grad_z);
                                         
    template<class T, typename F> 
      void advectSlabs(CCVariable<facedata<T> >& q_OAFS,
                       const Patch* patch,
                       const CCVariable<T>& q_CC,
                       CCVariable<T>& q_advected,
                       SFCXVariable<double>& q_XFC,
                       SFCYVariable<double>& q_YFC,
                       SFCZVariable<double>& q_ZFC,
                       F save_q_FC);  // passed in function
                       
    template<class T>
      void q_FC_operator(CellIterator iter, 
                         IntVector adj_offset,
                         const int face,
                         const CCVariable<facedata<double> >& q_OAFS,
                         const CCVariable<double>& q_CC,
                         T& q_FC);
                
      void q_FC_PlusFaces(const CCVariable<facedata<double> >& q_OAFS,
                          const CCVariable<double>& q_CC,
                          const Patch* patch,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC);
                          
    template<class T, class V>
      void q_FC_flux_operator(CellIterator iter, 
                              IntVector adj_offset,
                              const int face,
                              const Patch* patch,
                              const bool is_Q_massSpecific,
                              const CCVariable<facedata<V> >& q_OAFS,
                              T& q_FC_flux);
                                  
      template<class T>
        void q_FC_fluxes(const CCVariable<T>& q_CC,
                         const CCVariable<facedata<T> >& q_OAFS,
                         const string& desc,
                         advectVarBasket* vb);
         
  };  
} // end namespace Uintah

#endif


