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

#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>

#include <Core/Thread/Mutex.h>

#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/MDSystem.h>

#include <Core/Util/DebugStream.h>

#include <iostream>
#include <cmath>

#include <sci_values.h>

//using namespace Uintah;

namespace Uintah {

  SCIRun::Mutex MD_flowLock("MD flow control lock");

  static DebugStream mdSystemDebug("MDSystemDebug", false);

  MDSystem::MDSystem()
  {

  }

  MDSystem::~MDSystem()
  {

  }

  MDSystem::MDSystem(const ProblemSpecP&    ps,
                           GridP&           grid,
                           Forcefield*      _ff)
                    :d_forcefield(_ff)
  {
    //std::vector<int> d_atomTypeList;
    f_atomsRegistered = false;
    ProblemSpecP mdsystem_ps = ps->findBlock("MD")->findBlock("System");
    std::string ensembleLabel;

    if (!mdsystem_ps) {
      throw ProblemSetupException(
          "Could not find \"System\" subsection of MD block.",
          __FILE__,
          __LINE__);
    }
    mdsystem_ps->getAttribute("ensemble",ensembleLabel);
    if ("NVE" == ensembleLabel) {
      d_ensemble = NVE;
    }
    else if ("NVT" == ensembleLabel) {
      d_ensemble = NVT;
      mdsystem_ps->require("temperature", d_temperature);
    }
    else if ("NPT" == ensembleLabel) {
      d_ensemble = NPT;
      mdsystem_ps->require("temperature", d_temperature);
      mdsystem_ps->require("pressure", d_pressure);
    }
    else if ("Isokinetic" == ensembleLabel) {
      d_ensemble = ISOKINETIC;
      mdsystem_ps->require("temperature", d_temperature);
    }
    else { // Unknown ensemble listed
      std::stringstream errorOut;
      errorOut << "ERROR in the System section of the MD specification block!"
               << std::endl
               << "  Unknown ensemble requested: " << ensembleLabel << std::endl
               << "  Available ensembles are:  \"NVE  NVT   NPT   Isokinetic\""
               << std::endl;
      throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
    }

    if (mdSystemDebug.active()) {
      mdSystemDebug << "MD::MDSystem | "
                    << "Parsed ensemble information" << std::endl;
    }

    d_numAtoms      =   0;
    d_numMolecules  =   0;

    // Reference the atomMap here to get number of atoms of Type into system

  }

  size_t MDSystem::registerAtomTypes(const atomMap*             incomingMap,
                                     const SimulationStateP&    simState)
  {
//    if (!f_atomsRegistered)
//    {
//      MD_flowLock.lock();
//      f_atomsRegistered = true;
      d_numAtoms = 0;
      d_numAtomsOfType.clear();
      size_t numTypes = incomingMap->getNumberAtomTypes();
      for (size_t currType = 0; currType < numTypes; ++currType) {
        std::string atomLabel;
        atomLabel = simState->getMDMaterial(currType)->getMaterialLabel();
        size_t numOfType = incomingMap->getAtomListSize(atomLabel);
        d_numAtoms += numOfType;
        d_numAtomsOfType.push_back(numOfType);
      }
//      MD_flowLock.unlock();
//    }
    return d_numAtoms;
  }

  size_t MDSystem::registerAtomCount(const size_t count,
                                     const size_t matlIndex)
  {
//    MD_flowLock.lock();
    size_t numMatls = d_numAtomsOfType.size();
    if (matlIndex < numMatls) {
      d_numAtomsOfType[matlIndex] = count;
      d_numAtoms += count;
    };
//    MD_flowLock.unlock();
    return count;
//        size_t numAtomTypes = getNumAtomTypes();

  }


} // namespace Uintah_MD
