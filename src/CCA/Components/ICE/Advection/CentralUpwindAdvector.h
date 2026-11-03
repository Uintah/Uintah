/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef UINTAH_CENTRAL_UPWIND_ADVECTOR_H
#define UINTAH_CENTRAL_UPWIND_ADVECTOR_H

#include <CCA/Components/ICE/Advection/Advector.h>
#include <CCA/Components/ICE/Core/CentralUpwind.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Disclosure/TypeDescription.h>

namespace Uintah {

  //______________________________________________________________________
  //  Kurganov-Tadmor central-upwind advection operator.  The per-face
  //  weights (alphaOwn, aSf, kappa) and mass fluxes (phiOwn, phiNei) are
  //  precomputed once per timestep by ICE::computeCentralWeights and
  //  ICE::computeCentralFluxes and carried into advectMass/advectQ through
  //  advectVarBasket (hasCentralFluxes + the six phiOwn/phiNei pointers).
  //  Method: ICE/PIMPLE_KT_method.pdf, Section 6.  Math: Core/CentralUpwind.h.
  class CentralUpwindAdvector : public Advector {

  public:
    CentralUpwindAdvector();
    explicit CentralUpwindAdvector( centralUpwind::Limiter limiter );
    CentralUpwindAdvector( centralUpwind::Limiter limiter,
                           DataWarehouse* new_dw,
                           const Patch* patch,
                           const bool isNewGrid );

    virtual ~CentralUpwindAdvector();

    virtual CentralUpwindAdvector* clone( DataWarehouse* new_dw,
                                          const Patch* patch,
                                          const bool isNewGrid );

    virtual void inFluxOutFluxVolume(const SFCXVariable<double>& uvel_FC,
                                     const SFCYVariable<double>& vvel_FC,
                                     const SFCZVariable<double>& wvel_FC,
                                     const double& delT,
                                     const Patch* patch,
                                     const int&  indx,
                                     const bool& bulletProof_test,
                                     DataWarehouse* new_dw,
                                     advectVarBasket* vb);

    virtual void  advectQ(const CCVariable<double>& q_CC,
                          const Patch* patch,
                          CCVariable<double>& q_advected,
                          advectVarBasket* vb,
                          SFCXVariable<double>& q_XFC,
                          SFCYVariable<double>& q_YFC,
                          SFCZVariable<double>& q_ZFC,
                          DataWarehouse* new_dw);

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

    centralUpwind::Limiter d_limiter{ centralUpwind::Limiter::vanAlbada };

    // Set by clone(): the patch this instance advects on, plus the level
    // cell-index range/periodicity used by the own/nei stencil-validity
    // test (ktStencilFlags in the .cc).  hi is exclusive; lo/hi already
    // include the extra-cell layer (Level::findCellIndexRange).
    const Patch* d_patch{ nullptr };
    IntVector    d_levelLo{ IntVector(0,0,0) };
    IntVector    d_levelHi{ IntVector(0,0,0) };
    IntVector    d_periodic{ IntVector(0,0,0) };

    // Set by inFluxOutFluxVolume().  The pointers refer to the caller's
    // face velocities and are valid only for the task that set them.
    double d_delT{ 0.0 };
    Vector d_dx{ Vector(0,0,0) };
    const SFCXVariable<double>* d_uvel_FC{ nullptr };
    const SFCYVariable<double>* d_vvel_FC{ nullptr };
    const SFCZVariable<double>* d_wvel_FC{ nullptr };

    void checkNoRefluxing( const advectVarBasket* vb ) const;

    template<class T>
      void advectQ_massSpecific( const CCVariable<T>&   q_CC,
                                 const CCVariable<double>& mass,
                                 CCVariable<T>&          q_advected,
                                 advectVarBasket*        vb );
  };

} // Uintah namespace

#endif
