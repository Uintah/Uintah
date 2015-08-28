/*
 *
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
 *
 * ----------------------------------------------------------
 * EwaldRealspace.h
 *
 *  Created on: Jul 9, 2015
 *      Author: jbhooper
 */

#ifndef EWALDREALSPACE_H_
#define EWALDREALSPACE_H_

#include <CCA/Components/MD/Electrostatics/Electrostatics.h>

namespace Uintah {
  class EwaldRealspace   {
    public:
      static void realspaceTholeDipole(
                                       const ProcessorGroup*     pg,
                                       const PatchSubset*        patches,
                                       const MaterialSubset*     atomTypes,
                                             DataWarehouse*      subOldDW,
                                             DataWarehouse*      subNewDW,
                                             DataWarehouse*      parentOldDW,
                                       const SimulationStateP*   simState,
                                       const MDLabel*            label,
                                       const double&             ewaldBeta,
                                       const double&             electrostaticCutoff,
                                       const int&                electrostaticGhostCells
                                      );
      static void realspacePointDipole(
                                              const ProcessorGroup*     pg,
                                              const PatchSubset*        patches,
                                              const MaterialSubset*     atomTypes,
                                                    DataWarehouse*      subOldDW,
                                                    DataWarehouse*      subNewDW,
                                                    DataWarehouse*      parentOldDW,
                                              const SimulationStateP*   simState,
                                              const MDLabel*            label,
                                              const double&             ewaldBeta,
                                              const double&             electrostaticCutoff,
                                              const int&                electrostaticGhostCells
                                             );
      static void realspaceChargeOnly(
                                             const ProcessorGroup*      pg,
                                             const PatchSubset*         patches,
                                             const MaterialSubset*      atomTypes,
                                                   DataWarehouse*       subOldDW,
                                                   DataWarehouse*       subNewDW,
                                                   DataWarehouse*       parentOldDW,
                                             const SimulationStateP*    simState,
                                             const MDLabel*             label,
                                             const double&              ewaldBeta,
                                             const double&              electrostaticCutoff,
                                             const int&                 electrostaticGhostCells
                                            );

     static void generatePointScreeningMultipliers(
                                                    const double& radius,
                                                    const double& beta,
                                                          double& B0,
                                                          double& B1,
                                                          double& B2,
                                                          double& B3
                                                  );
     static void generateTholeScreeningMultipliers(
                                                    const double& a_thole,
                                                    const double& sqrt_ai_aj,
                                                    const double& r,
                                                          double& B1,
                                                          double& B2,
                                                          double& B3
                                                  );

     static void generateDipoleFunctionalTerms(
                                                const double&           q_i,
                                                const double&           q_j,
                                                const SCIRun::Vector&   mu_i,
                                                const SCIRun::Vector&   mu_j,
                                                const SCIRun::Vector&   r_ij,
                                                      double&           mu_jDOTr_ij,
                                                      double&           G0,
                                                      double&           G1_mu_q,
                                                      double&           G1_mu_mu,
                                                      double&           G2,
                                                      SCIRun::Vector&   gradG0,
                                                      SCIRun::Vector&   gradG1,
                                                      SCIRun::Vector&   gradG2
                                              );
  };
}

#endif /* EWALDREALSPACE_H_ */
