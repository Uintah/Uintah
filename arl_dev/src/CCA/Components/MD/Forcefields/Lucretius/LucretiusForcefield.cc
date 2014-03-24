/*
 * LucretiusForcefield.cc
 *
 *  Created on: Mar 14, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/MDMaterial.h>
#include <CCA/Components/MD/Forcefields/parseUtils.h>
#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusForcefield.h>
#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusMaterial.h>
#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusParsing.h>
#include <CCA/Components/MD/Potentials/TwoBody/TwoBodyPotentialFactory.h>

#include <Core/Grid/SimulationStateP.h>
#include <Core/Malloc/Allocator.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <fstream>
#include <iostream>

using namespace Uintah;

const std::string LucretiusForcefield::d_forcefieldNameString = "Lucretius";

NonbondedTwoBodyPotential* LucretiusForcefield::parseHomoatomicNonbonded(std::string& parseLine,
                                                                   const forcefieldType currentForcefieldType,
                                                                   double currentMass) {
// First three characters form the label for this specific interaction type
  std::string nbLabel;
  nbLabel = parseLine.substr(0,3);  // First three characters form the specific interaction label

  std::vector<std::string> token_list;
  Parse::tokenizeAtMost(parseLine.substr(3,std::string::npos),token_list,6);

// Token map:  A = 0, B = 1, C = 2, D = 3, Mass = 4, potential type = 5, comments = rest
  size_t numTokens = token_list.size();
  if (numTokens < 6) { // We should always have this info no matter what
    std::ostringstream errorOut;
    errorOut << "ERROR in Lucretius forcefield file potential definition." << std::endl
             << "  Line as input: >> " << std::endl
             << "    "  << parseLine << std::endl
             << "FILE INFO: ";
    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
  }

  currentMass = Parse::stringToDouble(token_list[4]);

  std::string potentialType = token_list[5];
  std::string fullComment = "";
  if (numTokens > 6) {
    fullComment = token_list[7];
  }

// Create our potential
  NonbondedTwoBodyPotential* potential;
  potential = NonbondedTwoBodyFactory::create(currentForcefieldType, potentialType, token_list, nbLabel, fullComment);
  return potential;
}

bool LucretiusForcefield::skipComments(std::ifstream& fileHandle, std::string& buffer) {

  if (!fileHandle) return false; // file was EOF on entry

  getline(fileHandle,buffer); // Prime the buffer
  if (!fileHandle && (buffer[0] != '*')) return true; // The single line exhausted our buffer, but we have a valid line

  while (fileHandle) {
    if (buffer[0] == '*') {
      getline(fileHandle,buffer);
    }
    else {
      return true; // Found a non-comment line
    }
  }
  return false; // Ran out of file
}

void LucretiusForcefield::generateUnexpectedEOFString(const std::string& filename,
                                                      const std::string& addendum,
                                                      std::string& buffer) {
  std::stringstream errorBuffer;
  errorBuffer << "ERROR:  Unexpected end of file " << filename << std::endl
              << "   -- Unable to locate the " << addendum << " section of the Lucretius forcefield file. ";
  buffer = errorBuffer.str();
}

void LucretiusForcefield::parseNonbondedPotentials(std::ifstream& fileHandle,
                                                   const std::string& filename,
                                                   std::string& buffer,
                                                   SimulationStateP& sharedState) {
// ---> Number of homoatomic potentials
  forcefieldType currentForcefieldType = Lucretius;
  size_t numberNonbondedTypes = 0;
  std::string error_msg;
  if (skipComments(fileHandle,buffer)) { // Locate the number of homo-atomic nonbonded potentials
    numberNonbondedTypes = Parse::stringToInt(buffer);
  }
  else {
    this->generateUnexpectedEOFString(filename,"NUMBER OF HETEROATOMIC REPULSION-DISPERSION POTENTIALS",error_msg);
    throw ProblemSetupException(error_msg, __FILE__, __LINE__);
  }

// ---> Definition of homoatomic Potentials
  if (skipComments(fileHandle,buffer)) { // Locate the section which defines nonbonded homoatomic potentials, charges, and polarizabilities
    double currentMass = -1.0;
    size_t numberNonbondedFound = 0;
    size_t currentChargeSubindex = 0;
    NonbondedTwoBodyPotential* currentNonbonded;
    while (buffer[0] != '*') { // Parse until we hit a new comment section
      getline(fileHandle,buffer);
      if (buffer[0] != ' ') { // Found a nonbonded type
        currentNonbonded = parseHomoatomicNonbonded(buffer, currentForcefieldType, currentMass);
        ++numberNonbondedFound;
        currentChargeSubindex = 1;
      }
      else { // Line starts with a space so represents a charge / polarization / comment
        std::vector<std::string> chargeLineTokens;
        Parse::tokenizeAtMost(buffer, chargeLineTokens, 2); // return 2 tokens and the rest of the line with leading spaces stripped
        double charge = Parse::stringToInt(chargeLineTokens[0]);
        double polarizability = Parse::stringToInt(chargeLineTokens[1]);
        std::string chargeLineComment = "";
        if (chargeLineTokens.size() == 3) {
          chargeLineComment = chargeLineTokens[2];
        }
        MDMaterial* currentMaterial = scinew LucretiusMaterial(currentNonbonded,
                                                               currentMass,
                                                               charge,
                                                               polarizability,
                                                               currentChargeSubindex);
        ++currentChargeSubindex;
        sharedState->registerMDMaterial(currentMaterial);
        nonbondedTwoBodyKey potentialKey(currentNonbonded->getLabel(),currentNonbonded->getLabel());
        potentialMap.insert(twoBodyPotentialMapPair(potentialKey,currentNonbonded));
      }
    }
    if (numberNonbondedFound != numberNonbondedTypes) {
      std::ostringstream error_stream;
      error_stream << "ERROR:  Expected " << numberNonbondedTypes << " homoatomic nonbonded potentials." << std::endl
                   << "  However, found " << numberNonbondedFound << " homoatomic nonbonded potentials." << std::endl;
      throw ProblemSetupException(error_stream.str(), __FILE__, __LINE__);
    }
  } // end of nonbonded homoatomic section
  else {
      this->generateUnexpectedEOFString(filename,"HOMOATOMIC REPULSION-DISPERSION AND CHARGE DEFINITIONS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
  }
// ---> Definition of heteroatomic Potentials
  if (skipComments(fileHandle,buffer)) { // Locate the section which defines heteroatomic potentials
  while (buffer[0] != '*') { // Found a heteroatomic nonbonded type
    std::string label1 = buffer.substr(0,3);
    std::string label2 = buffer.substr(4,3);


  }
  }
  else {
    this->generateUnexpectedEOFString(filename,"HETEROATOMIC REPULSION-DISPERSION DEFINITIONS", error_msg);
    throw ProblemSetupException(error_msg, __FILE__, __LINE__);
  }
  //if (skipComments)
}

LucretiusForcefield::LucretiusForcefield(const ProblemSpecP& spec,
                                         SimulationStateP& sharedState) {

  ProblemSpecP ffspec = spec->findBlock("MD")->findBlock("Forcefield");
  forcefieldType currentForcefieldType = Lucretius;


  if (ffspec) {
    // Parse the forcefield file for the potentials within the file.
    // Materials will be registered from within the forcefield parsing, so we need
    //   to pass in the shared state.
    std::string ffFilename;
    ffspec->get("forcefield_file", ffFilename);
    std::ifstream ffFile;
    ffFile.open(ffFilename.c_str(), std::ifstream::in);
    if (!ffFile) {
      throw ProblemSetupException("Could not open the Lucretius forcefield input file.", __FILE__ , __LINE__);
    }

//-> BEGIN PARSING FORCEFIELD
    std::string buffer;
    std::string error_msg;



// ---> Definition of heteroatomic potentials
    if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
    // Parse hetero-atomic potentials here
    }
    else {
      this->generateUnexpectedEOFString(ffFilename,"HETEROATOMIC REPULSION-DISPERSION DEFINITIONS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
// ---> Number of bond potentials
    size_t numberBondTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      numberBondTypes = Parse::stringToInt(buffer);
    }
    else {
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF BOND POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberBondTypes != 0) {
      if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      // !FIXME Parse bond types here
      }
      else {
        this->generateUnexpectedEOFString(ffFilename,"BOND POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> Number of bend potentials
    size_t numberBendTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      numberBendTypes = Parse::stringToInt(buffer);
    }
    else {
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF BEND POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberBendTypes != 0) {
      if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      // !FIXME Parse bond types here
      }
      else {
        this->generateUnexpectedEOFString(ffFilename,"BEND POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> Number of dihedral potentials
    size_t numberTorsionTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      numberTorsionTypes = Parse::stringToInt(buffer);
    }
    else {
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF DIHEDRAL POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberTorsionTypes != 0) {
      if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      // !FIXME Parse bond types here
      }
      else {
        this->generateUnexpectedEOFString(ffFilename,"DIHEDRAL POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> Number of OOP potentials
    size_t numberOOPTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      numberOOPTypes = Parse::stringToInt(buffer);
    }
    else {
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF IMPROPER DIHEDRAL (OOP) POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberOOPTypes != 0) {
      if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      // !FIXME Parse bond types here
      }
      else {
        this->generateUnexpectedEOFString(ffFilename,"IMPROPER DIHEDRAL (OOP) POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> Number of lone pair (LP) types
    size_t numberLPTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      numberLPTypes = Parse::stringToInt(buffer);
    }
    else {
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF LONE PAIR TYPES",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberOOPTypes != 0) {
      if (skipComments(ffFile,buffer)) { // Locate the section which defines hetero-atomic nonbonded potentials
      // !FIXME Parse bond types here
      }
      else {
        this->generateUnexpectedEOFString(ffFilename,"LONE PAIR DESCRIPTIONS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }

  }
  else
  {
    throw ProblemSetupException("Could not find the Forcefield block in the input file.", __FILE__, __LINE__);
  }
}



//    while (ffFile) {
//        size_t numberNonbondedFound = 0;
//        NonbondedTwoBodyPotential* currentNonbonded;
//        double currentMass = -1.0;
//        size_t currentChargeSubindex = 0;
//
//
//        while (numberNonbondedFound < numberNonbondedTypes) {
//          getline(ffFile,buffer);
//          std::stringstream inputStream(buffer);
//          while (buffer [0] != '*') { // In nonbonded homoatomic sections
//            if (buffer[0] != ' ') { // Found a nonbonded type
//              std::string lucretiusLabel = buffer.substr(0,3);
//              currentNonbonded = this->parseNonbondedType(buffer, currentForcefieldType, currentMass);
//              ++numberNonbondedFound;
//              currentChargeSubindex = 1;
//            }
//            else if (buffer[0] != '*') { // Found a new charge type for current nonbonded type
//              double charge, polarizability;
//              std::string comment;
//              std::istringstream inputBuffer(buffer);
//              inputBuffer >> charge >> polarizability >> comment;
//              // Once we have charge/polarizability and a nonbonded potential it's time to make a material
//              MDMaterial* currentMaterial = scinew LucretiusMaterial(currentNonbonded,
//                                                                     currentMass,
//                                                                     charge,
//                                                                     polarizability,
//                                                                     currentChargeSubindex);
//              sharedState->registerMDMaterial(currentMaterial);
//            }
//          } // end of nonbonded homoatomic section
//          if (numberNonbondedFound != numberNonbondedTypes) {
//            std::stringstream errorBuffer;
//            errorBuffer << "ERRORS:  Expected " << numberNonbondedTypes << " nonbonded potentials, but found "
//                        <<  numberNonbondedFound << "!";
//            throw ProblemSetupException(errorBuffer.str(), __FILE__, __LINE__);
//          }
//        }
//        else
//        while (buffer[0] == '*') {
//          getline(ffFile,buffer);
//        }
//        while
//
//
//
//        // Questions for Alan:
//        //
//        //  NOTE:  How to register particle sets with materials
//        //   1)  Read particle coordinates
//        //   2)  Sort particles into appropriate patches
//        //   3)  Sort particles into groups by atom type
//        //   4)  Create particle subset with material corresponding to atom type
//        //   5)  Uintah will handle interleaving each of these particles from there
//
//        //   Register materials from here okay?
//        //   Nonbonded potentials:  Do I need to explicity add reference trackers to keep them alive?
//        //   Task Graphs embedded in components (future facing re: integrator)
//        //   Add reference to material?
//        //   Why would a material register particle state?
//
//        // End of nonbonded types
//      } // found all nonbonded
//    } // end of file
//  } // if (ffspec) [forcefield block found]
//  else
//  {
//    throw ProblemSetupException("Cannot locate the name of the Lucretius forcefield file to parse.", __FILE__, __LINE__);
//  }
//}
//


