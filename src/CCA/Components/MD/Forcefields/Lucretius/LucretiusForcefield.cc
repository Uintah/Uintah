/*
 * LucretiusForcefield.cc
 *
 *  Created on: Mar 14, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusForcefield.h>

#include <CCA/Components/MD/Forcefields/parseUtils.h>

#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusParsing.h>

#include <CCA/Components/MD/Potentials/TwoBody/TwoBodyPotentialFactory.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidState.h>

#include <Core/Util/DebugStream.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace Uintah;
/*
 * ....................................................................................................................*
 */

// Debug streams for FF parsing
static DebugStream lucretiusDebug("LucretiusFFParse", false);

const std::string LucretiusForcefield::d_forcefieldNameString = "Lucretius";


NonbondedTwoBodyPotential*
LucretiusForcefield::parseHomoatomicNonbonded(std::string&         parseLine,
                                              const forcefieldType currFFType,
                                              double&              currentMass)
{

  // First three characters form the label for this specific interaction type
  std::string nbLabel;
  nbLabel = parseLine.substr(0,3);

  std::vector<std::string> token_list;
  Parse::tokenizeAtMost(parseLine.substr(3,std::string::npos),token_list,6);
// Token map:  A = 0, B = 1, C = 2, D = 3, Mass = 4, potential type = 5,
//             Remainder of line -> Comments
  size_t numTokens = token_list.size();

  if (numTokens < 6) { // We are required to have the first 6 tokens
    std::ostringstream errorOut;
    errorOut    << "ERROR in Lucretius forcefield file potential definition."
                << std::endl
                << "  Line as input: >> " << std::endl
                << "    "  << parseLine << std::endl
                << "FILE INFO: ";
    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
  }

  currentMass = Parse::stringToDouble(token_list[4])*this->ffMassToInternal();
  std::string potentialType = token_list[5];
  std::string fullComment = "";

  if (numTokens > 6) { // Anything left over is a comment
    fullComment = token_list[6];
  }

// Create our potential
  NonbondedTwoBodyPotential* potential;
  potential = NonbondedTwoBodyFactory::create(this,
                                              potentialType,
                                              token_list,
                                              nbLabel,
                                              fullComment);
  return potential;
}

NonbondedTwoBodyPotential*
LucretiusForcefield::parseHeteroatomicNonbonded(std::string&         parseLine,
                                                const forcefieldType currFFType)
{
  // Heteroatomic potentials are denoted by two homoatomic labels
  // joined with "_" stored in the first string, and followed by
  // A, B, C and D parameters for the cross-term interaction

  // Extract the label string seperately
  std::string nbLabel;
  nbLabel = parseLine.substr(0,7);
  std::vector<std::string> token_list;

  // Parse A, B, C, D from the remaining substring
  Parse::tokenizeAtMost(parseLine.substr(7,std::string::npos),token_list,4);
  size_t numTokens = token_list.size();

  if (numTokens < 4) { // A, B, C, D should always be present.
    std::ostringstream errorOut;
    errorOut << "ERROR in Lucretius forcefield file potential definition."
             << std::endl
             << "  Line as input: >> " << std::endl
             << "    " << parseLine << std::endl
             << "FILE INFO: ";
    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
  }

  std::string fullComment = "";
  if (numTokens == 5) { // Anything else is a comment
    fullComment = token_list[5];
  }

  // Lucretius doesn't define the requisite interaction types explicitly in
  // cross-interactions. Instead, it tends to regard them implicitly:
  //    A non-zero B term implies LucretiusExp6, otherwise LJ
  //    The only wrinkle is technically it's possible for that LJ to be 12-6
  //    or 9-6 under Lucretius, even though I don't believe there's a FF out
  //    there with a 9-6 potential.

  NonbondedTwoBodyPotential* potential;
  std::string potentialType = "";

  //  Determine potential type.
  double BValue = Parse::stringToDouble(token_list[1]);
  if (abs(BValue) >= 1.0e-12) { // Real B value, use exp-6
    potentialType = "exp-6";
  }
  else { // We're using LJ functions of some type, determine their type
    std::string label1 = nbLabel.substr(0,3);
    std::string label2 = nbLabel.substr(4,3);
    NonbondedTwoBodyPotential* potential1;
    potential1 = potentialMap.find(nonbondedTwoBodyKey(label1,label1))->second;
    NonbondedTwoBodyPotential* potential2;
    potential2 = potentialMap.find(nonbondedTwoBodyKey(label2,label2))->second;
    std::string potentialString1 = potential1->getPotentialDescriptor();
    std::string potentialString2 = potential2->getPotentialDescriptor();
    if (potentialString1 != potentialString2) {
      // Can't mix two potentials of differing types
      std::ostringstream errorOut;
      errorOut << "ERROR in Lucretius forcefield heteroatomic potential section."
               << std::endl
               << "  There is no prescribed way to determine the potential type "
               << "for the mixed potential using labels: "
               << std::endl << "     "
               << label1 << " of type [" << potentialString1 << "] with "
               << label2 << " of type [" << potentialString2 << "]." << std::endl
               << "  Original input line: " << std::endl
               << parseLine << std::endl
               << "FILE INFO: ";
      throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
    }
    else { // Potentials of the same type, either LJ 9-6 or LJ 12-6
      if (potentialString1.find("_12-6") != std::string::npos) {
        potentialType = "lj126";
      }
      else if (potentialString1.find("_9-6") != std::string::npos) {
        potentialType = "lj9-6";
      }
      else { // These potentials aren't one strict Lucretius understands
        std::ostringstream errorOut;
        errorOut << "ERROR in Lucretius forcefield heteroatomic potential"
                 << " section."
                 << std::endl
                 << "  Potentials for the interaction of atom type: "
                 << std::endl
                 << "     "  << label1 << " of type [" << potentialString1 << "]"
                 << std::endl
                 << "     "  << label2 << " of type [" << potentialString2 << "]"
                 << std::endl
                 << "  Match, but are not recognized as Lucretius forcefield"
                 << " potentials."
                 << std::endl
                 << "Original input line: \"" << parseLine << "\"" << std::endl
                 << "FILE INFO: ";
        throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
      }
    }
  }

  potential = NonbondedTwoBodyFactory::create(this,
                                              potentialType,
                                              token_list,
                                              nbLabel,
                                              fullComment);
  return potential;
}

void LucretiusForcefield::parseNonbondedPotentials(std::ifstream&       inStream,
                                                   const std::string&   inName,
                                                   std::string&         buffer,
                                                   SimulationStateP&    simState)
{
// ---> Number of homoatomic potentials
  forcefieldType currentForcefieldType = Lucretius;
  size_t numberNonbondedTypes = 0;
  std::string error_msg;
  if (lucretiusParse::skipComments(inStream,buffer))
  { // Number of homo-atomic nonbonded potentials
    numberNonbondedTypes = Parse::stringToInt(buffer);
  }
  else { // Error in FF file: Number of nonbonded potentials
    lucretiusParse::generateUnexpectedEOFString(inName,
      "NUMBER OF HETEROATOMIC REPULSION-DISPERSION POTENTIALS",
      error_msg);
    throw ProblemSetupException(error_msg, __FILE__, __LINE__);
  }

  if (lucretiusParse::skipComments(inStream,buffer)) // Parse nonbonded
  { // Find nonbonded potentials, charges, and polarizabilities
    double currentMass = -1.0;
    size_t numberNonbondedFound = 0;
    size_t currentChargeSubindex = 0;
    NonbondedTwoBodyPotential* nonbondedHomoatomic;
    while (buffer[0] != '*') { // Parse until we hit a new comment section
      if (buffer[0] != ' ' && buffer[0] != '\t') { // Found a nonbonded type
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
      else { // Line starts with space: represents charge/polarization/comment
        std::vector<std::string> chargeLineTokens;
        Parse::tokenizeAtMost(buffer, chargeLineTokens, 2);
        // Charge = 0, Polarizability = 1, comment = 2
        double charge = Parse::stringToDouble(chargeLineTokens[0]);
        double polarizability = Parse::stringToDouble(chargeLineTokens[1]);
        std::string chargeLineComment = "";
        if (chargeLineTokens.size() == 3) { // Store comment
          chargeLineComment = chargeLineTokens[2];
        }
        LucretiusMaterial* currentMaterial;
        currentMaterial = scinew LucretiusMaterial(nonbondedHomoatomic,
                                                   currentMass,
                                                   charge,
                                                   polarizability,
                                                   currentChargeSubindex);
        materialArray.push_back(currentMaterial);
        if (lucretiusDebug.active()) { // Debug material registration
          size_t index = materialArray.size() - 1;
          lucretiusDebug << "MD::LucretiusForcefield | "
                         << "  Material " << index << " of type "
                         << materialArray[index]->getMaterialDescriptor()
                         << " parsed as \""
                         << materialArray[index]->getMaterialLabel()
                         << "\"" << std::endl;
        }
        ++currentChargeSubindex;
        nonbondedTwoBodyKey potentialKey(nonbondedHomoatomic->getLabel(),
                                         nonbondedHomoatomic->getLabel());
        potentialMap.insert(twoBodyPotentialMapPair(potentialKey,
                                                    nonbondedHomoatomic));
      }
      getline(inStream,buffer);
    }
    if (numberNonbondedFound != numberNonbondedTypes)
    {   // Verify expected number of potentials
      std::ostringstream error_stream;
      error_stream << "ERROR:  Expected " << numberNonbondedTypes
                   << " homoatomic nonbonded potentials." << std::endl
                   << "  However, found " << numberNonbondedFound
                   << " homoatomic nonbonded potentials." << std::endl;
      throw ProblemSetupException(error_stream.str(), __FILE__, __LINE__);
    }
  } // end of nonbonded homoatomic section
  else { // Couldn't find nonbonded section
    std::string errorText;
    errorText = "HOMOATOMIC REPULSION-DISPERSION AND CHARGE DEFINITIONS";
    lucretiusParse::generateUnexpectedEOFString(inName, errorText, error_msg);
    throw ProblemSetupException(error_msg, __FILE__, __LINE__);
  }
// ---> End of homoatomic potentials
// ---> Definition of heteroatomic Potentials
  if (lucretiusParse::skipComments(inStream,buffer)) // Parse cross-terms
  { // Locate the heteroatomic potential section
    while (buffer[0] != '!') { // Found a heteroatomic nonbonded type
      std::string label1 = buffer.substr(0,3);
      std::string label2 = buffer.substr(4, 3);
      // Ensure the base material potentials exist
      nonbondedTwoBodyKey key1(label1, label1);
      nonbondedTwoBodyKey key2(label2, label2);
      nonbondedTwoBodyMapType::iterator location1, location2;
      if (   potentialMap.count(key1) != 0
          && potentialMap.count(key2) != 0 )
      { // Corresponds to homoatomic types
        NonbondedTwoBodyPotential* crossTerm;
        crossTerm = parseHeteroatomicNonbonded(buffer, currentForcefieldType);
        if (crossTerm) { // Heteroatomic potential successfully created
          nonbondedTwoBodyKey crossKey(label1, label2);
          potentialMap.insert(twoBodyPotentialMapPair(crossKey, crossTerm));
          // Also insert the potential on the reversed key for ease of look up later.
          // U(a,b) == U(b,a) for any potentials we'll be interested in, at
          // least at the 2 body level of theory
          nonbondedTwoBodyKey reverseKey(label2, label1);
          potentialMap.insert(twoBodyPotentialMapPair(reverseKey, crossTerm));
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
        errorOut << "Could not find a base potential for one or more "
                 << "heteroatomic potentials." << std::endl
                 << "  Input line: \"" << buffer << "\"" << std::endl;
        throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
      }
      getline(inStream,buffer);
    }
  }
  else {  // EOF before end of heteroatomic potential inputs
    lucretiusParse::generateUnexpectedEOFString(inName,"HETEROATOMIC REPULSION-DISPERSION DEFINITIONS", error_msg);
    throw ProblemSetupException(error_msg, __FILE__, __LINE__);
  }
  //  Now let's double check that we have definitions for all possible heteroatomic potentials
  int numHomoatomic = materialArray.size();
//  int numHomoatomic = simState->getNumMDMatls();
  if (lucretiusDebug.active())
  {
    lucretiusDebug << "MD::LucretiusForcefield | "
                   << "Found " << numHomoatomic
                   << " different charge types." << std::endl;
  }
  // Check to make sure we have explicit definition of all cross terms
  for (int index1 = 0; index1 < numHomoatomic; ++index1)
  {
    std::string label1;
    label1 = materialArray[index1]->getPotentialHandle()->getLabel();
    for (int index2 = 0; index2 < numHomoatomic; ++index2)
    {
      std::string label2;
      label2 = materialArray[index2]->getPotentialHandle()->getLabel();

      nonbondedTwoBodyKey mapKey(label1,label2);
      if (potentialMap.find(mapKey) == potentialMap.end()) {
        std::ostringstream errorOut;
        errorOut << "ERROR in Lucretius forcefield!"
                 << "  Potentials for both " << "\"" << label1 << "\""
                 << " and " << "\"" << label2 << "\""
                 << " are present, but the cross-potential "
                 << "\"" << label1 << "_" << label2 << "\""
                 << " is not." << std::endl
                 << "  At this point, implicit definition of cross terms is"
                 << " not supported in this forcefield type." << std::endl;
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
      // If we got here without throwing an error, we have:
      //   a)  The right number of homoatomic potentials
      //   b)  A heteroatomic potential for every homoatomic potential
      //       combination
      // This should represent a complete forcefield
    }
  }
  if (lucretiusDebug.active()) {
    lucretiusDebug << "MD::LucretiusForcefield | "
                   << "All cross terms explicitly specified." << std::endl;
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
    ffspec->require("forcefieldFile", ffFilename);
    std::ifstream ffFile;
    ffFile.open(ffFilename.c_str(), std::ifstream::in);
    if (!ffFile) {
      std::stringstream errorOut;
      errorOut << "ERROR:  Could not open Lucretius forcefield input file: \""
               << ffFilename << "\"" << std::endl << "\t";
      throw ProblemSetupException(errorOut.str(), __FILE__ , __LINE__);
    }

//-> BEGIN PARSING FORCEFIELD
    std::string buffer;
    std::string error_msg;

    // Parse the nonbonded potentials
    parseNonbondedPotentials(ffFile, ffFilename, buffer, sharedState);

// ---> Number of bond potentials
    size_t numberBondTypes = 0;
    if (lucretiusParse::skipComments(ffFile,buffer)) { // Num bond potentials
      numberBondTypes = Parse::stringToInt(buffer);
    }
    else { // Num bond potentials not found
      lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                  "NUMBER OF BOND POTENTIALS",
                                                  error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberBondTypes != 0) { // --> Bonds indicated, so parse them
      if (lucretiusParse::skipComments(ffFile,buffer)) {// Bond potential block
      // parseBondPotentials(ffFile, ffFilename, buffer);
      }
      else { // Bond potential block not found
        lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                    "BOND POTENTIALS",
                                                    error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of bond potential parsing

// ---> Number of bend potentials
    size_t numberBendTypes = 0;
    if (lucretiusParse::skipComments(ffFile,buffer)) { // Num bend potentials
      numberBendTypes = Parse::stringToInt(buffer);
    }
    else { // Num bend potentials not found
      lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                  "NUMBER OF BEND POTENTIALS",
                                                  error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberBendTypes != 0) {        // --> Bends indicated, so parse them
      if (lucretiusParse::skipComments(ffFile,buffer)) { // Bend potential block
      // parseBendPotentials(ffFile, ffFilename, buffer);
      }
      else {                               // EOF before bend potentials
        lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                    "BEND POTENTIALS",
                                                    error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of Bend potential parsing

// ---> Number of Torsional potentials
    size_t numberTorsionTypes = 0;
    if (lucretiusParse::skipComments(ffFile,buffer)) { // Num dihedral potentials
      numberTorsionTypes = Parse::stringToInt(buffer);
    }
    else { // Num dihedral potentials not found
      lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                  "NUMBER OF DIHEDRAL POTENTIALS",
                                                  error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberTorsionTypes != 0) {     // --> Torsions indicated, so parse them
      if (lucretiusParse::skipComments(ffFile,buffer)) { // dihedral potentials
      // parseTorsionalPotentials(ffFile, ffFilename, buffer);
      }
      else { // dihedral potentials not found
        lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                    "DIHEDRAL POTENTIALS",
                                                    error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of Torsion parsing

// ---> Number of OOP potentials
    size_t numberOOPTypes = 0;
    if (lucretiusParse::skipComments(ffFile,buffer)) { // Num OOP potentials
      numberOOPTypes = Parse::stringToInt(buffer);
    }
    else { // Num out-of-plane (OOP) potentials not found
      std::string errorText;
      errorText = "NUMBER OF IMPROPER DIHEDRAL (OOP) POTENTIALS";
      lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                  errorText,
                                                  error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberOOPTypes != 0) {         // --> OOP indicated, so parse them
      if (lucretiusParse::skipComments(ffFile,buffer)) { // OOP potentials
      // parseOOPPotentials(ffFile, ffFilename, buffer);
      }
      else { // OOP potentials not found
        std::string errorText;
        errorText = "IMPROPER DIHEDRAL (OOP) POTENTIALS";
        lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                    errorText,
                                                    error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of OOP parsing

// ---> Number of lone pair (LP) types
    size_t numberLPTypes = 0;
    if (lucretiusParse::skipComments(ffFile,buffer)) { // Num LP types
      numberLPTypes = Parse::stringToInt(buffer);
    }
    else { // Num LP types not found
      lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                  "NUMBER OF LONE PAIR TYPES",
                                                  error_msg);
      throw ProblemSetupException(error_msg, __FILE__, __LINE__);
    }
    if (numberOOPTypes != 0) {         // --> Lone pairs indicated, parse them
      if (lucretiusParse::skipComments(ffFile,buffer)) { // Lone pair descript
      // parseLPTypes(ffFile, ffFilename, buffer);
      }
      else { // Lone pair descriptions
        lucretiusParse::generateUnexpectedEOFString(ffFilename,
                                                    "LONE PAIR DESCRIPTIONS",
                                                    error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
// ---> End of lone pair parsing

  }
  else {  // Couldn't find the forcefield block in the PS
    throw ProblemSetupException(
        "Could not find the Forcefield block in the input file.",
        __FILE__,
        __LINE__);
  }
  if (lucretiusDebug.active()) {
    lucretiusDebug << "MD::LucretiusForcefield | Built Lucretius Forcefield"
                   << std::endl;
  }
}

NonbondedTwoBodyPotential*
LucretiusForcefield::getNonbondedPotential(const std::string& label1,
                                           const std::string& label2) const
{
  nonbondedTwoBodyMapType::const_iterator potentialLocation;

  nonbondedTwoBodyKey mapKey(label1,label2);
  potentialLocation = potentialMap.find(mapKey);

  if (potentialLocation != potentialMap.end()) {
    return potentialLocation->second;
  }
  std::ostringstream ErrorOut;
  ErrorOut << "ERROR:  Attempt to index an unreferenced nonbonded potential:"
           << std::endl
           << "  Label1:  " << label1 << std::endl
           << "  Label2:  " << label2 << std::endl;
  throw InvalidState(ErrorOut.str(), __FILE__, __LINE__);
}

void LucretiusForcefield::registerAtomTypes(
    const varLabelArray& states_IterationN,
    const varLabelArray& states_IterationNPlusOne,
    const MDLabel* label,
    SimulationStateP& simState) const {

  size_t numberAtomTypes = materialArray.size();

  for (size_t atomType = 0; atomType < numberAtomTypes; ++atomType) {
    simState->registerMDMaterial(materialArray[atomType]);
    simState->d_cohesiveZoneState.push_back(states_IterationN);
    simState->d_cohesiveZoneState_preReloc.push_back(states_IterationNPlusOne);
    if (lucretiusDebug.active()) {
      lucretiusDebug << "MD::LucretiusForcefield |  "
                     << "Registered material " << atomType << " -> \""
                     << materialArray[atomType]->getMaterialLabel() << "\""
                     << std::endl;
    }
  }


}

