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
/*
 * ....................................................................................................................*
 */
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
  potential = NonbondedTwoBodyFactory::create(currentForcefieldType,
                                              potentialType,
                                              token_list,
                                              nbLabel,
                                              fullComment);
  return potential;
}

NonbondedTwoBodyPotential* LucretiusForcefield::parseHeteroatomicNonbonded(std::string& parseLine,
                                                                           const forcefieldType currentForcefieldType)
{
  std::string nbLabel;
  nbLabel = parseLine.substr(0,7);  // Heteroatomic potentials are denoted by two homoatomic labels joined with "_"
  std::vector<std::string> token_list;

  Parse::tokenizeAtMost(parseLine.substr(7,std::string::npos),token_list,4); // Only A, B, C, D for cross interactions

  size_t numTokens = token_list.size();
  if (numTokens < 4) { // A = 0; B = 1; C = 2; D = 3
    std::ostringstream errorOut;
    errorOut << "ERROR in Lucretius forcefield file potential definition." << std::endl
             << "  Line as input: >> " << std::endl
             << "    " << parseLine << std::endl
             << "FILE INFO: ";
    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
  }

  std::string fullComment = "";
  if (numTokens == 5) { // Comment is present
    fullComment = token_list[5];
  }

  // Lucretius doesn't define the requisite interaction types explicitly in cross-interactions.
  //   Instead, it tends to regard them implicitly:  A non-zero B term implies LucretiusExp6, otherwise LJ
  //   The only wrinkle is technically it's possible for that LJ to be 12-6 or 9-6 under Lucretius, even though
  //   I don't believe there's a FF out there with a 9-6 potential.

  NonbondedTwoBodyPotential* potential;
  std::string potentialType = "";
  //  Determine potential type.
  double BValue = Parse::stringToDouble(token_list[1]);
  if (abs(BValue) >= 1.0e-12) { // Real B value, use exp-6
    potentialType = "exp-6";
  }
  else { // We're using LJ functions of some type
    std::string label1 = nbLabel.substr(0,3);
    std::string label2 = nbLabel.substr(4,3);
    NonbondedTwoBodyPotential* potential1 = potentialMap.find(nonbondedTwoBodyKey(label1,label1))->second;
    NonbondedTwoBodyPotential* potential2 = potentialMap.find(nonbondedTwoBodyKey(label2,label2))->second;
    std::string potentialString1 = potential1->getPotentialDescriptor();
    std::string potentialString2 = potential2->getPotentialDescriptor();
    if (potentialString1 != potentialString2) {
      std::ostringstream errorOut;
      errorOut << "ERROR in Lucretius forcefield heteroatomic potential section." << std::endl
               << "  There is no prescribed way to determine the potential type for the mixed potential using labels: "
               << std::endl << "     "
               << label1 << " of type [" << potentialString1 << "] with "
               << label2 << " of type [" << potentialString2 << "]." << std::endl
               << "  Original input line: " << std::endl
               << parseLine << std::endl
               << "FILE INFO: ";
      throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
    }
    else { // Potentials of the same type, either LJ 9-6 or LJ 12-6
      if (potentialString1.find("_12-6") != std::string::npos) potentialType = "lj126";
      if (potentialString1.find("_9-6") != std::string::npos)  potentialType = "lj9-6";
      if (potentialType == "") {
        std::ostringstream errorOut;
        errorOut << "ERROR in Lucretius forcefield heteroatomic potential section." << std::endl
                 << "  Potentials for the interaction of atom type: " << std::endl
                 << "     "  << label1 << " of type [" << potentialString1 << "]" << std::endl
                 << "     "  << label2 << " of type [" << potentialString2 << "]" << std::endl
                 << "  Match, but are not recognized as Lucretius forcefield potentials." << std::endl
                 << "Original input line: " << std::endl << parseLine << std::endl
                 << "FILE INFO: ";
        throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
      }
    }
  }

  potential = NonbondedTwoBodyFactory::create(currentForcefieldType,
                                              potentialType,
                                              token_list,
                                              nbLabel,
                                              fullComment);
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
    NonbondedTwoBodyPotential* nonbondedHomoatomic;
    while (buffer[0] != '*') { // Parse until we hit a new comment section
      getline(fileHandle,buffer);
      if (buffer[0] != ' ') { // Found a nonbonded type
        nonbondedHomoatomic = parseHomoatomicNonbonded(buffer, currentForcefieldType, currentMass);
        if (nonbondedHomoatomic) { // Created homoatomic potential
          ++numberNonbondedFound;
          currentChargeSubindex = 1;
        }
        else { // Could not create homoatomic potential
          std::ostringstream errorOut;
          errorOut << "Could not create potential for Homoatomic interaction:" << std::endl
                   << "  Input line: " << std::endl
                   << buffer << std::endl;
          throw ProblemSetupException(errorOut.str(),__FILE__,__LINE__);
        }
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
        MDMaterial* currentMaterial = scinew LucretiusMaterial(nonbondedHomoatomic,
                                                               currentMass,
                                                               charge,
                                                               polarizability,
                                                               currentChargeSubindex);
        ++currentChargeSubindex;
        sharedState->registerMDMaterial(currentMaterial);
        nonbondedTwoBodyKey potentialKey(nonbondedHomoatomic->getLabel(),nonbondedHomoatomic->getLabel());
        potentialMap.insert(twoBodyPotentialMapPair(potentialKey,nonbondedHomoatomic));
      }
    }
    if (numberNonbondedFound != numberNonbondedTypes) { // Check for expected number of potentials
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
// ---> End of homoatomic potentials

// ---> Definition of heteroatomic Potentials
  if (skipComments(fileHandle,buffer)) { // Locate the section which defines heteroatomic potentials
    while (buffer[0] != '*') { // Found a heteroatomic nonbonded type
      std::string label1 = buffer.substr(0,3);
      std::string label2 = buffer.substr(4, 3);
      // Ensure the base material potentials exist
      nonbondedTwoBodyKey key1(label1, label1);
      nonbondedTwoBodyKey key2(label2, label2);
      nonbondedTwoBodyMapType::iterator location1, location2;
      if (potentialMap.count(key1) != 0 && potentialMap.count(key2) != 0) { // Corresponds to homoatomic types
        NonbondedTwoBodyPotential* nonbondedHeteroatomic = parseHeteroatomicNonbonded(buffer, currentForcefieldType);
        if (nonbondedHeteroatomic) { // Successfully created potential, so insert it into our map
          nonbondedTwoBodyKey heteroatomicKey(label1, label2);
          potentialMap.insert(twoBodyPotentialMapPair(heteroatomicKey, nonbondedHeteroatomic));
          // Also insert the potential on the reversed key for ease of look up later.  U(a,b) == U(b,a) for any potentials
          //   we'll be interested in, at least at the 2 body level of theory
          nonbondedTwoBodyKey reverseHeteroatomicKey(label2, label1);
          potentialMap.insert(twoBodyPotentialMapPair(reverseHeteroatomicKey, nonbondedHeteroatomic));
        }
        else { // Couldn't create potential, throw exception
          std::ostringstream errorOut;
          errorOut << "Could not create potential for Heteroatomic interaction:" << std::endl
                   << "  Input line: "
                   << std::endl
                   << buffer
                   << std::endl;
          throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
        }
      }
      else { // Does not correspond to homoatomic types
        std::ostringstream errorOut;
        errorOut << "Could not find a base potential for one or more heteroatomic potentials." << std::endl
                 << "  Input line: "
                 << std::endl
                 << buffer
                 << std::endl;
        throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
      }
    }
  }
  else {  // EOF before end of heteroatomic potential inputs
    this->generateUnexpectedEOFString(filename,"HETEROATOMIC REPULSION-DISPERSION DEFINITIONS", error_msg);
    throw ProblemSetupException(error_msg, __FILE__, __LINE__);
  }
  //  Now let's double check that we have definitions for all possible heteroatomic potentials
  int numHomoatomic = sharedState->getNumMDMatls();
  for (int index1 = 0; index1 < numHomoatomic; ++index1) {
    std::string label1 = sharedState->getMDMaterial(index1)->getPotentialHandle()->getLabel();
    for (int index2 = 0; index2 < numHomoatomic; ++index2) {
      std::string label2 = sharedState->getMDMaterial(index2)->getPotentialHandle()->getLabel();
      nonbondedTwoBodyKey mapKey(label1,label2);
      if (potentialMap.find(mapKey) == potentialMap.end()) {
        std::ostringstream errorOut;
        errorOut << "ERROR in Lucretius forcefield!"
                 << "  Potentials for both " << "\"" << label1 << "\"" << " and "
                 << "\"" << label2 << "\"" << " are present, but the cross-potential "
                 << "\"" << label1 << "_" << label2 << "\"" << " is not." << std::endl
                 << "  At this point, implicit definitions of cross terms are not supported in this forcefield type."
                 << std::endl;
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
      // If we got here without throwing an error, we have:
      //   a)  The right number of homoatomic potentials
      //   b)  A heteroatomic potential for every homoatomic potential combination
      // This should represent a complete forcefield
    }
  }
// ---> End of heteroatomic potentials
  return;
}

LucretiusForcefield::LucretiusForcefield(const ProblemSpecP& spec,
                                         SimulationStateP& sharedState) {

  ProblemSpecP ffspec = spec->findBlock("MD")->findBlock("Forcefield");
  forcefieldType currentForcefieldType = Lucretius;

  if (ffspec) {  // Forcefield block found
    // Parse the forcefield file for the potentials within the file.

    // Read file name and open the required forcefield file (Lucretius format)
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

    // Parse the nonbonded potentials
    parseNonbondedPotentials(ffFile, ffFilename, buffer, sharedState);

// ---> Number of bond potentials
    size_t numberBondTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate the section with expected number of Bond potentials
      numberBondTypes = Parse::stringToInt(buffer);
    }
    else {                             // EOF before number of bond potentials found
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF BOND POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberBondTypes != 0) {        // --> Bonds indicated, so parse them
      if (skipComments(ffFile,buffer)) {   // Locate the section which describes bond potentials
      // parseBondPotentials(ffFile, ffFilename, buffer);
      }
      else {                               // EOF before bond potentials
        this->generateUnexpectedEOFString(ffFilename,"BOND POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of bond potential parsing

// ---> Number of bend potentials
    size_t numberBendTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate the section with expected number of Bend potentials
      numberBendTypes = Parse::stringToInt(buffer);
    }
    else {                             // EOF before number of bend potentials found
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF BEND POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberBendTypes != 0) {        // --> Bends indicated, so parse them
      if (skipComments(ffFile,buffer)) {   // Locate the section which describes bend potentials
      // parseBendPotentials(ffFile, ffFilename, buffer);
      }
      else {                               // EOF before bend potentials
        this->generateUnexpectedEOFString(ffFilename,"BEND POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of Bend potential parsing

// ---> Number of Torsional potentials
    size_t numberTorsionTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate section with expected number of torsional potentials
      numberTorsionTypes = Parse::stringToInt(buffer);
    }
    else {                             // EOF before number of torsional potentials found
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF DIHEDRAL POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberTorsionTypes != 0) {     // --> Torsions indicated, so parse them
      if (skipComments(ffFile,buffer)) {   // Locate the section which describes torsional potentials
      // parseTorsionalPotentials(ffFile, ffFilename, buffer);
      }
      else {                               // EOF before torsional potentials
        this->generateUnexpectedEOFString(ffFilename,"DIHEDRAL POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of Torsion parsing

// ---> Number of OOP potentials
    size_t numberOOPTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate section with number of OOP potentials
      numberOOPTypes = Parse::stringToInt(buffer);
    }
    else {                             // EOF before reading number of OOP potentials
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF IMPROPER DIHEDRAL (OOP) POTENTIALS",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberOOPTypes != 0) {         // --> OOP potentials indicated, so parse them
      if (skipComments(ffFile,buffer)) {   // Find the OOP potential descriptions
      // parseOOPPotentials(ffFile, ffFilename, buffer);
      }
      else {                               // EOF before OOP potentials
        this->generateUnexpectedEOFString(ffFilename,"IMPROPER DIHEDRAL (OOP) POTENTIALS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of OOP parsing

// ---> Number of lone pair (LP) types
    size_t numberLPTypes = 0;
    if (skipComments(ffFile,buffer)) { // Locate section with number of expected LP
      numberLPTypes = Parse::stringToInt(buffer);
    }
    else {                             // EOF before reading number of lone pairs
      this->generateUnexpectedEOFString(ffFilename,"NUMBER OF LONE PAIR TYPES",error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberOOPTypes != 0) {         // --> Lone pairs indicated, so parse them
      if (skipComments(ffFile,buffer)) {   // Locate the section which describes lone pairs
      // parseLPTypes(ffFile, ffFilename, buffer);
      }
      else {                               // EOF before Lone Pair description
        this->generateUnexpectedEOFString(ffFilename,"LONE PAIR DESCRIPTIONS",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of lone pair parsing
  }
  else {  // Couldn't find the forcefield block in the PS
    throw ProblemSetupException("Could not find the Forcefield block in the input file.", __FILE__, __LINE__);
  }
}
