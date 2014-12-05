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
 * MDSubordinate.cc
 *
 *  Created on: Nov 13, 2014
 *      Author: jbhooper
 */

#include <Core/Grid/DbgOutput.h>

#include <CCA/Components/MD/MD.h>

using namespace Uintah;

void MD::integratorInitialize(const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     atomTypes,
                                    DataWarehouse*      oldDW,
                                    DataWarehouse*      newDW)
{
  const std::string location            = "MD::integratorInitialize";
  const std::string flowLocation        = location + " | ";
  const std::string particleLocation    = location + " P ";
  printTask(patches, md_cout, location);

  d_integrator->initialize(pg, patches, atomTypes, oldDW, newDW,
                           &d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::integratorSetup(     const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     atomTypes,
                                    DataWarehouse*      oldDW,
                                    DataWarehouse*      newDW)
{
  const std::string location            = "MD::integratorSetup";
  const std::string flowLocation        = location + " | ";
  const std::string particleLocation    = location + " P ";
  printTask(patches, md_cout, location);

  d_integrator->setup(pg, patches, atomTypes, oldDW, newDW,
                      &d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::integratorCalculate( const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     atomTypes,
                                    DataWarehouse*      oldDW,
                                    DataWarehouse*      newDW)
{
  const std::string location            = "MD::integratorCalculate";
  const std::string flowLocation        = location + " | ";
  const std::string particleLocation    = location + " P ";
  printTask(patches, md_cout, location);

  d_integrator->calculate(pg, patches, atomTypes, oldDW, newDW,
                          &d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::integratorFinalize(  const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     atomTypes,
                                    DataWarehouse*      oldDW,
                                    DataWarehouse*      newDW)
{
  const std::string location            = "MD::integratorFinalize";
  const std::string flowLocation        = location + " | ";
  const std::string particleLocation    = location + " P ";
  printTask(patches, md_cout, location);

  d_integrator->finalize(pg, patches, atomTypes, oldDW, newDW,
                         &d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }

}

void MD::nonbondedInitialize(const ProcessorGroup*  pg,
                             const PatchSubset*     patches,
                             const MaterialSubset*  matls,
                             DataWarehouse*         oldDW,
                             DataWarehouse*         newDW)
{
  const std::string location = "MD::nonbondedInitialize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, "MD::nonbondedInitialize");

  d_nonbonded->initialize(pg, patches, matls, oldDW, newDW,
                          d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }


}

void MD::nonbondedSetup(const ProcessorGroup*   pg,
                        const PatchSubset*      patches,
                        const MaterialSubset*   matls,
                        DataWarehouse*          oldDW,
                        DataWarehouse*          newDW)
{
  const std::string location = "MD::nonbondedSetup";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_nonbonded->setup(pg, patches, matls, oldDW, newDW,
                     d_sharedState, d_system, d_label, d_coordinate);

//  if (d_coordinate->queryCellChanged()) {
//    d_nonbonded->setup(pg, patches, matls, oldDW, newDW,
//                       d_sharedState, d_system, d_label, d_coordinate);
//  }

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }


}

void MD::nonbondedCalculate(const ProcessorGroup*   pg,
                            const PatchSubset*      patches,
                            const MaterialSubset*   matls,
                            DataWarehouse*          oldDW,
                            DataWarehouse*          newDW)
{
  const std::string location = "MD::nonbondeCalculate";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, "MD::nonbondedCalculate");

  d_nonbonded->calculate(pg, patches, matls, oldDW, newDW,
                         d_sharedState, d_system, d_label, d_coordinate);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::nonbondedFinalize(const ProcessorGroup*    pg,
                           const PatchSubset*       patches,
                           const MaterialSubset*    matls,
                           DataWarehouse*           oldDW,
                           DataWarehouse*           newDW)
{
  const std::string location = "MD::nonbondedFinalize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_nonbonded->finalize(pg, patches, matls, oldDW, newDW,
                        d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsInitialize(const ProcessorGroup* pg,
                                  const PatchSubset*    patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*        oldDW,
                                  DataWarehouse*        newDW)
{
  const std::string location = "MD::electrostaticsInitialize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_electrostatics->initialize(pg, patches, matls, oldDW, newDW,
                               &d_sharedState, d_system, d_label, d_coordinate);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsSetup(const ProcessorGroup*  pg,
                             const PatchSubset*     patches,
                             const MaterialSubset*  matls,
                             DataWarehouse*         oldDW,
                             DataWarehouse*         newDW)
{
  const std::string location = "MD::electrostaticsSetup";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, "MD::electrostaticsSetup");

  d_electrostatics->setup(pg, patches, matls, oldDW, newDW,
                          &d_sharedState, d_system, d_label, d_coordinate);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsCalculate(const ProcessorGroup*  pg,
                                 const PatchSubset*     perProcPatches,
                                 const MaterialSubset*  matls,
                                 DataWarehouse*         parentOldDW,
                                 DataWarehouse*         parentNewDW,
                                 const LevelP           level)
{
  const std::string location = "MD::electrostaticsCalculate";
  const std::string flowLocation = location + " | ";
  const std::string electrostaticLocation = location + " E ";
  printTask(perProcPatches, md_cout, location);

  // Copy del_t to the subscheduler
//  delt_vartype dt;
//  DataWarehouse* subNewDW = d_electrostaticSubscheduler->get_dw(3);
//  parentOldDW->get(dt,
//                   d_sharedState->get_delt_label(),
//                   level.get_rep());
//  subNewDW->put(dt,
//                d_sharedState->get_delt_label(),
//                level.get_rep());

  if (electrostaticDebug.active()) {
    electrostaticDebug << electrostaticLocation
                       << "  Copied delT to the electrostatic subscheduler."
                       << std::endl;
  }

  d_electrostatics->calculate(pg,perProcPatches,matls,parentOldDW,parentNewDW,
                              &d_sharedState, d_system, d_label, d_coordinate,
                              d_electrostaticSubscheduler, level);

  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}

void MD::electrostaticsFinalize(const ProcessorGroup*   pg,
                                const PatchSubset*      patches,
                                const MaterialSubset*   matls,
                                DataWarehouse*          oldDW,
                                DataWarehouse*          newDW)
{
  const std::string location = "MD::electrostaticsFinalize";
  const std::string flowLocation = location + " | ";
  printTask(patches, md_cout, location);

  d_electrostatics->finalize(pg, patches, matls, oldDW, newDW,
                             &d_sharedState, d_system, d_label, d_coordinate);
  if (mdFlowDebug.active()) {
    mdFlowDebug << flowLocation
                << "END"
                << std::endl;
  }
}



