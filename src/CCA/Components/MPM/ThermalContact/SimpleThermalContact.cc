/*
 * SimpleThermalContact.cc
 *
 *  Created on: May 3, 2018
 *      Author: jbhooper
 */
#include <CCA/Components/MPM/ThermalContact/SimpleThermalContact.h>

namespace Uintah
{
  SimpleThermalContact::SimpleThermalContact( ProblemSpecP      & _inSpec
                                            , SimulationStateP  & _inManager
                                            , MPMLabel          * _inMPMLabel
                                            , MPMFlags          * _inFlags
                                            )
  {
    m_mpmLbl          = _inMPMLabel;
    m_materialManager = _inManager;
    m_mpmFlags        = _inFlags;

    m_gThermalInterfaceFlag = VarLabel::create("g.thermalInterfaceFlag",
                                            NCVariable<int>::getTypeDescription());
    if (m_mpmFlags->d_fracture) {
      m_GThermalInterfaceFlag = VarLabel::create("G.thermalInterfaceFlag",
                                              NCVariable<int>::getTypeDescription());

    }

  }

  SimpleThermalContact::~SimpleThermalContact()
  {
    VarLabel::destroy(m_gThermalInterfaceFlag);
    if (m_GThermalInterfaceFlag != nullptr) {
      VarLabel::destroy(m_GThermalInterfaceFlag);
    }
  }

  void SimpleThermalContact::addComputesAndRequires(      Task        * task
                                                   ,const PatchSet    * patches
                                                   ,const MaterialSet * materials )
                                                   const
  {

    Ghost::GhostType gan    = Ghost::AroundNodes;
    Ghost::GhostType gnone  = Ghost::None;
    task->requires(Task::OldDW, m_mpmLbl->delTLabel);

    task->requires(Task::NewDW, m_mpmLbl->gMassLabel,        Ghost::None);
    task->requires(Task::NewDW, m_mpmLbl->gTemperatureLabel, Ghost::None);
    task->computes(m_mpmLbl->gThermalContactTemperatureRateLabel);
    task->computes(m_gThermalInterfaceFlag);

    task->computes(m_mpmLbl->gThermalContactTemperatureRateLabel,
                   m_materialManager->getAllInOneMatl(), Task::OutOfDomain);
    task->computes(m_gThermalInterfaceFlag, m_materialManager->getAllInOneMatl(),
                   Task::OutOfDomain);

    if (m_mpmFlags->d_fracture) {
      // Second field for fracture
      task->requires(Task::NewDW, m_mpmLbl->GMassLabel, Ghost::None);
      task->requires(Task::NewDW, m_mpmLbl->GTemperatureLabel, Ghost::None);
      task->computes(m_mpmLbl->GThermalContactTemperatureRateLabel);
      task->computes(m_GThermalInterfaceFlag);
      task->computes(m_mpmLbl->GThermalContactTemperatureRateLabel,
                  m_materialManager->getAllInOneMatl(), Task::OutOfDomain);
      task->computes(m_GThermalInterfaceFlag, m_materialManager->getAllInOneMatl(),
                  Task::OutOfDomain);
    }
  }

  void SimpleThermalContact::computeHeatExchange(const  ProcessorGroup  *
                                                ,const  PatchSubset     * patches
                                                ,const  MaterialSubset  * materials
                                                ,       DataWarehouse   * old_dw
                                                ,       DataWarehouse   * new_dw    )
  {
    int numMats = materials->size();

    Ghost::GhostType  typeGhost;
    int               numGhost;
    m_materialManager->getParticleGhostLayer(typeGhost, numGhost);

    delt_vartype delT;
    old_dw->get(delT, m_mpmLbl->delTLabel, getLevel(patches));
    double delTInv = 1.0/delT;

    for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx)
    {
      const Patch * patch     = patches->get(patchIdx);
      int           totalDWI  = m_materialManager->getAllInOneMatl()->get(0);

      // Constant global variables for the default mass field
      constNCVariable<double>               gTotalMass;
      new_dw->get(gTotalMass, m_mpmLbl->gMassLabel, totalDWI, patch, typeGhost, numGhost);
      // Temporary global variables for the default mass field
      NCVariable<double>                gTotalThermalMass;
      NCVariable<double>                gdTdt_interface_total;
      NCVariable<int>                   gdTdt_interface_flag_global;

      new_dw->allocateTemporary(gTotalThermalMass, patch);
      gTotalThermalMass.initialize(0.0);              // Total nodal thermal mass
      new_dw->allocateAndPut(gdTdt_interface_total,
                             m_mpmLbl->gThermalContactTemperatureRateLabel,
                             totalDWI, patch);
      gdTdt_interface_total.initialize(0.0);          // Total nodal rate
      new_dw->allocateAndPut(gdTdt_interface_flag_global, m_gThermalInterfaceFlag,
                             totalDWI, patch);
      gdTdt_interface_flag_global.initialize(false);  // Global interface flag

      // Per material constant variables for the default mass field
      std::vector<constNCVariable<double> > gMass(numMats);
      std::vector<constNCVariable<double> > gTemperature(numMats);
      // Calculated per-material reponse for the default mass field
      std::vector<NCVariable<double> >  gdTdt_interface(numMats);
      std::vector<NCVariable<int> >     gdTdt_interface_flag(numMats);
      std::vector<NCVariable<double> >  gThermalMass(numMats);

      // FRACTURE Related ----  Fixme todo There's got to be a cleaner way - JBH
      // Constant global variables for the fracture mass field
      constNCVariable<double>               GTotalMass;
      // Temporary global variables for the fracture mass field
      NCVariable<double>                GTotalThermalMass;
      NCVariable<double>                GdTdt_interface_total;
      NCVariable<int>                   GdTdt_interface_flag_global;
      // Per material constant variables for the fracture mass field
      std::vector<constNCVariable<double> > GMass(numMats);
      std::vector<constNCVariable<double> > GTemperature(numMats);
      // Calculated per-material response for the fracture mass field
      std::vector<NCVariable<int> >     GdTdt_interface_flag(numMats);
      std::vector<NCVariable<double> >  GThermalMass(numMats);
      std::vector<NCVariable<double> >  GdTdt_interface(numMats);

      if (m_mpmFlags->d_fracture) {
        new_dw->get(GTotalMass, m_mpmLbl->GMassLabel, totalDWI, patch, typeGhost, numGhost);
        new_dw->allocateTemporary(GTotalThermalMass, patch);
        GTotalThermalMass.initialize(0.0);
        new_dw->allocateAndPut(GdTdt_interface_total,
                               m_mpmLbl->GThermalContactTemperatureRateLabel,
                               totalDWI, patch);
        GdTdt_interface_total.initialize(0.0);
        new_dw->allocateAndPut(GdTdt_interface_flag_global,
                               m_GThermalInterfaceFlag,
                               totalDWI, patch);
        GdTdt_interface_flag_global.initialize(false);
      } // Load constant and allocate temporary global variables for fracture mass field
      // FRACTURE Related ----

      for (int matIdx = 0; matIdx < numMats; ++matIdx) {
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*> (m_materialManager->getMPMMaterial(matIdx));
        int dwi = mpm_matl->getDWIndex();
        const double specificHeat = mpm_matl->getSpecificHeat();

        new_dw->get(gMass[matIdx], m_mpmLbl->gMassLabel,
                    dwi, patch, typeGhost, numGhost);
        new_dw->get(gTemperature[matIdx], m_mpmLbl->gTemperatureLabel,
                    dwi, patch, typeGhost, numGhost);

        new_dw->allocateAndPut(gdTdt_interface[matIdx],
                               m_mpmLbl->gThermalContactTemperatureRateLabel,
                               dwi, patch, typeGhost, numGhost);
        gdTdt_interface[matIdx].initialize(0);

        new_dw->allocateTemporary(gThermalMass[matIdx], patch);
        gThermalMass[matIdx].initialize(0);

        new_dw->allocateAndPut(gdTdt_interface_flag[matIdx],
                               m_gThermalInterfaceFlag,
                               dwi, patch, typeGhost, numGhost);
        gdTdt_interface_flag[matIdx].initialize(false);

        for (NodeIterator nIt = patch->getNodeIterator(); !nIt.done(); ++nIt) {
          IntVector node = *nIt;
          double thermalMass = gMass[matIdx][node]*specificHeat;
          gThermalMass[matIdx][node] = thermalMass;
          gTotalThermalMass[node] += thermalMass;
        } // Iterate over nodes for all normal values

        if (m_mpmFlags->d_fracture) {
          new_dw->get(GMass[matIdx], m_mpmLbl->GMassLabel,
                      dwi, patch, typeGhost, numGhost);
          new_dw->get(GTemperature[matIdx], m_mpmLbl->GTemperatureLabel,
                      dwi, patch, typeGhost, numGhost);

          new_dw->allocateTemporary(GThermalMass[matIdx], patch);
          GThermalMass[matIdx].initialize(0.0);

          new_dw->allocateAndPut(GdTdt_interface[matIdx],
                                 m_mpmLbl->GThermalContactTemperatureRateLabel,
                                 dwi, patch, typeGhost, numGhost);
          GdTdt_interface[matIdx].initialize(0);

          new_dw->allocateAndPut(GdTdt_interface_flag[matIdx],
                                 m_GThermalInterfaceFlag,
                                 dwi, patch, typeGhost, numGhost);
          GdTdt_interface_flag[matIdx].initialize(false);


          for (NodeIterator nIt = patch->getNodeIterator(); !nIt.done(); ++nIt) {
            IntVector node = *nIt;
            double fractureThermalMass = GMass[matIdx][node]*specificHeat;
            GThermalMass[matIdx][node] = fractureThermalMass;
            GTotalThermalMass[node]   += fractureThermalMass;
          } // Iterate over nodes for all fracture field values
        } // If d_fracture
      } // Iterate over materials

      // Determine nodes on an interface
      const double minPresence = 1e-100; // Min mass for material to be considered present
      for (NodeIterator nIt = patch->getNodeIterator(); !nIt.done(); ++nIt) {
        IntVector node            = *nIt;
        int       materialIndex   = 0;
        bool      interfaceFound  = false;
        while ((materialIndex < numMats) && !interfaceFound) {
          double checkMass = gMass[materialIndex][node];
          if ((checkMass > minPresence) && checkMass != gTotalMass[node]) {
            interfaceFound = true;
          }
          ++materialIndex;
        } // Node checked
        // Tag interfrace on material grid for all materials present
        if (interfaceFound) {
          gdTdt_interface_flag_global[node] = true;
          for (materialIndex = 0; materialIndex < numMats; ++materialIndex) {
            if (gMass[materialIndex][node] > minPresence) {
              gdTdt_interface_flag[materialIndex][node] = true;
            }
          }
        }  // Interface found
        if (m_mpmFlags->d_fracture) {
          int       fractureIndex   = 0;
          bool      fractureInterfaceFound  = false;
          while ((fractureIndex < numMats) && !fractureInterfaceFound) {
            double checkMass = GMass[fractureIndex][node];
            if ((checkMass > minPresence) && checkMass != gTotalMass[node]) {
              fractureInterfaceFound = true;
            }
            ++fractureIndex;
          } // Node checked
          // Tag interfrace on material grid for all materials present
          if (fractureInterfaceFound) {
            GdTdt_interface_flag_global[node] = true;
            for (fractureIndex = 0; fractureIndex < numMats; ++fractureIndex) {
              if (GMass[fractureIndex][node] > minPresence) {
                GdTdt_interface_flag[fractureIndex][node] = true;
              }
            }
          }  // Interface found
        } // d_fracture
      } // Nodal interface search

      for (NodeIterator nIt = patch->getNodeIterator(); !nIt.done(); ++nIt) {
        double nodalHeat = 0.0;
        IntVector node = *nIt;

        if (gdTdt_interface_flag_global[node] == true) {
          for (int matIdx = 0; matIdx < numMats; ++matIdx) {
            if (gdTdt_interface_flag[matIdx][node] == true) {
              nodalHeat += gTemperature[matIdx][node]*gThermalMass[matIdx][node];
            } // Material at node has interface presence
          } // Loop over materials
          double contactTemp = nodalHeat/gTotalThermalMass[node];
          for (int matIdx = 0; matIdx < numMats; ++matIdx) {
            double dTdt = delTInv * (contactTemp - gTemperature[matIdx][node]);
            gdTdt_interface[matIdx][node] = dTdt;
            gdTdt_interface_total[node] += dTdt;
          }
        } // Node has interface
        double fractureHeat = 0.0;
        if (m_mpmFlags->d_fracture && GdTdt_interface_flag_global[node] == true) {
          for (int matIdx = 0; matIdx < numMats; ++matIdx) {
            if (GdTdt_interface_flag[matIdx][node] == true) {
              fractureHeat += GTemperature[matIdx][node]*GThermalMass[matIdx][node];
            } // Fracture material at node has interface presence
          } // Loop over fracture materials
          double fractureTemp = fractureHeat/GTotalThermalMass[node];
          for (int matIdx = 0; matIdx < numMats; ++matIdx) {
            double f_dTdt = delTInv * (fractureTemp - GTemperature[matIdx][node]);
            GdTdt_interface[matIdx][node] = f_dTdt;
            GdTdt_interface_total[node] += f_dTdt;
          }
        } // Fracture node has interface
      } // Loop over nodes
    } // Iterate over patches
  } // computeHeatExchange

  void SimpleThermalContact::initializeThermalContact(const Patch         * patch
                                                     ,      int          /* vfindex */
                                                     ,      DataWarehouse * new_dw    )
  {

  }

  void SimpleThermalContact::outputProblemSpec(ProblemSpecP & ps)
  {
    ProblemSpecP thermal_ps = ps->appendChild("thermal_contact");
    thermal_ps->appendElement("type","simple");
//    d_materials_list.outputProblemSpec(thermal_ps);

  }
}
