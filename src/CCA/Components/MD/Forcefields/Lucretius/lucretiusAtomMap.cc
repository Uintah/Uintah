/*
 * lucretiusAtomMap.cc
 *
 *  Created on: Mar 27, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>
#include <CCA/Components/MD/Forcefields/parseUtils.h>

#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusParsing.h>
#include <CCA/Components/MD/Forcefields/Lucretius/lucretiusAtomMap.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

#include <sstream>
#include <iomanip>

using namespace Uintah;

constLucretiusMapIterator
lucretiusAtomMap::findValidAtomList(const std::string& searchLabel) const
{
  constLucretiusMapIterator labelLocation = atomSet.find(searchLabel);

  if (labelLocation == atomSet.end()) { // Error:  Label not found
    std::stringstream errorOut;
    errorOut << "ERROR:  Attempt to access an atom type that does not exist."
             << std::endl
             << "  Forcefield Model --> Lucretius " << std::endl
             << "     Missing Label ---> \"" << searchLabel << "\"" << std::endl;
    throw InvalidValue(errorOut.str(), __FILE__, __LINE__);
    return (atomSet.end());
  }
  else return (labelLocation);
}

lucretiusMapIterator
lucretiusAtomMap::findValidAtomList(const std::string& searchLabel)
{
  lucretiusMapIterator labelLocation = atomSet.find(searchLabel);

  if (labelLocation == atomSet.end()) { // Error:  Label not found
    std::stringstream errorOut;
    errorOut << "ERROR:  Attempt to access an atom type that does not exist."
             << std::endl
             << "  Forcefield Model --> Lucretius " << std::endl
             << "     Missing Label ---> \"" << searchLabel << "\"" << std::endl;
    throw InvalidValue(errorOut.str(), __FILE__, __LINE__);
    return (atomSet.end());
  }
  else return (labelLocation);
}

size_t
lucretiusAtomMap::addAtomToList(const std::string&   searchLabel,
                                      atomData*      atomPtr)
{
  lucretiusMapIterator labelLocation = atomSet.find(searchLabel);

  if (labelLocation != atomSet.end())
  { // Label already exists, so add to its entry.
    labelLocation->second->push_back(atomPtr);
    return (labelLocation->second->size());
  }
  else
  { // Label not found, so create a new entry for it.
    std::vector<atomData*>* newAtomDataArray = new std::vector<atomData*>;
    std::pair<lucretiusMapIterator,bool> result;
    result = atomSet.insert(lucretiusMapPair(searchLabel,newAtomDataArray));

    if (result.second)
    { // New vector successfully created
      // result.first is the iterator, so result.first->second is the vector pointer
      (result.first)->second->push_back(atomPtr);
      return(1);  // We just placed the first element in the array
    }
    else
    { // Could not create the new vector.  There's been some type of error somewhere.
      std::stringstream errorOut;
      errorOut << "ERROR:  Could not create new list for atom type." << std::endl
               << "  Forcefield Model --> Lucretius " << std::endl
               << "        Atom Label ---> \"" << searchLabel << "\"" << std::endl;
      throw InternalError(errorOut.str(), __FILE__, __LINE__);
      return(0);
    }
  }
}

lucretiusAtomMap::lucretiusAtomMap(const ProblemSpecP&      spec,
                                   const SimulationStateP&  shared_state,
                                   const Forcefield*        forcefield)
{
  ProblemSpecP coordSpec = spec->findBlock("MD")->findBlock("Forcefield");
  SCIRun::Vector box, barostatX, barostatV, barostatA;
  std::vector<double> thermostatX, thermostatV, thermostatA;

  if (coordSpec) // Parse forcefield block
  {
    std::string ffType;
    coordSpec->getAttribute("type",ffType);

    if (ffType == "Lucretius") // All as expected
    {
      std::string coordFilename;
//      forcefieldType currentForcefieldType = Lucretius;
      coordSpec->require("coordinateFile", coordFilename);

      std::ifstream coordFile;
      coordFile.open(coordFilename.c_str(), std::ifstream::in);
      if (!coordFile) // Could not open the file
      {
        std::string ErrorMsg = "Could not open the Lucretius coordinate input file.";
        throw ProblemSetupException(ErrorMsg, __FILE__, __LINE__);
      }

      std::string buffer;
      std::string error_msg;
      forcefieldType ffType=Lucretius;
      if (lucretiusParse::skipComments(coordFile,buffer)) // Skip to data
      {
        SCIRun::Point   currentX;
        SCIRun::Vector  currentV;
        std::string     currentLabel;
        size_t          currentChargeIndex;
        long64          atomCount=0;

        // First data line is expected to be an atom position
        bool coordinates = true;
        double distanceConversion = forcefield->ffDistanceToInternal();
        double velocityConversion = forcefield->ffVelocityToInternal();
        double accelerationConversion = forcefield->ffAccelerationToInternal();

        while (buffer[0] != '*') // Parse coordinates and velocities
        {
          if (coordinates) // First parse the position and type label
          { // Current line is an atom position
            std::vector<std::string> parseTokens;
            Parse::tokenizeAtMost(buffer,parseTokens,5);
            if (parseTokens.size() < 5) // ERROR:  Not a complete line
            {
              std::stringstream errorOut;
              errorOut << "ERROR:  Parsed incomplete line where coordinate line"
                       << " was expected." << std::endl
                       << "  Forcefield Type:  Lucretius" << std::endl
                       << "  Parsed Line:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
            }
            currentX.x(Parse::stringToDouble(parseTokens[0]));
            currentX.y(Parse::stringToDouble(parseTokens[1]));
            currentX.z(Parse::stringToDouble(parseTokens[2]));

            // Convert units to simulation standard SI units
            currentX *= distanceConversion;

            currentLabel = parseTokens[3];  // Text portion of a Lucretius style atom label
            currentChargeIndex = static_cast<size_t> (Parse::stringToInt(parseTokens[4]));
            coordinates = false; // Next line should be velocity not coordinates
          }
          else // Then parse a velocity
          {
            std::vector<std::string> parseTokens;
            Parse::tokenizeAtMost(buffer,parseTokens,3);
            for (size_t index = 0; index < 3; ++index)
            {
              currentV[index] = Parse::stringToDouble(parseTokens[index]);
            }
            // Convert units to simulation standard SI units
            currentV *= velocityConversion;

            ++atomCount;
            // Have coordinates, velocity, and label, and ID# now.  Create atom data.
            lucretiusAtomData* newAtom = new lucretiusAtomData(currentX,
                                                               currentV,
                                                               atomCount,
                                                               currentLabel,
                                                               currentChargeIndex,
                                                               ffType);
            this->addAtomToList(newAtom->getLabel(),newAtom);
            coordinates=true;
          }
          getline(coordFile,buffer);
        }  // End of coordinate section

        // Parse extended variables
        if (lucretiusParse::skipComments(coordFile,buffer)) // Read box definition
        {
          std::vector<std::string> parseTokens;
          Parse::tokenize(buffer, parseTokens);
          if (parseTokens.size() != 3) {
            std::stringstream errorOut;
            errorOut << "ERROR: Box size should contain three numbers in Lucretius"
                     << " coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Box Line:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
          }
          // Save the box
          box = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                               Parse::stringToDouble(parseTokens[1]),
                               Parse::stringToDouble(parseTokens[2]));
          box *= forcefield->ffDistanceToInternal();
        }
        else  // Box info not found
        {
          lucretiusParse::generateUnexpectedEOFString(coordFilename,
                                                      "UNIT CELL DEFINITION",
                                                      error_msg);
          throw ProblemSetupException(error_msg, __FILE__, __LINE__);
        }

// Thermostat extended lagrangian variables
        if (lucretiusParse::skipComments(coordFile,buffer)) // Read thermostat pos.
        {
          std::vector<std::string> parseTokens;
          Parse::tokenize(buffer, parseTokens);
          size_t maxTokens = parseTokens.size();
          if (maxTokens < 1) { // Should be at least one thermostat variable here
            std::stringstream errorOut;
            errorOut << "ERROR:  No thermostat position variables found in "
                     << "expected line in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Thermostat Position Line:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          else // Store thermostat positions
          {
            for (size_t Index=0; Index < maxTokens; ++Index) {
              thermostatX.push_back(Parse::stringToDouble(parseTokens[Index])
                                    *distanceConversion);
            }
          }
        }
        else // Thermostat positions not found
        {
          lucretiusParse::generateUnexpectedEOFString(coordFilename,
                                                      "THERMOSTAT POSITION(S)",
                                                      error_msg);
          throw ProblemSetupException(error_msg, __FILE__, __LINE__);
        }
        if (lucretiusParse::skipComments(coordFile,buffer)) // Thermostat veloc.
        {
          std::vector<std::string> parseTokens;
          Parse::tokenize(buffer, parseTokens);
          size_t maxTokens = parseTokens.size();
          if (maxTokens < 1) // Should be at least one velocity
          {
            std::stringstream errorOut;
            errorOut << "ERROR:  No thermostat velocity variables found in expected line in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Thermostat Velocity:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          else if (maxTokens != thermostatX.size())
          {
            std::stringstream errorOut;
            errorOut << "ERROR:  Differing number of thermostat position and velocity variables found in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Thermostat Positions parsed: " << thermostatX.size() << std::endl
                     << "  Thermostat Velocities parsed: " << maxTokens << std::endl
                     << "  Velocity input buffer:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          else // Store thermostat velocities
          {
            for (size_t Index=0; Index < maxTokens; ++Index) {
              thermostatV.push_back(Parse::stringToDouble(parseTokens[Index])
                                    * velocityConversion);
            }
          }
        }
        else // Thermostat velocities not found
        {
          lucretiusParse::generateUnexpectedEOFString(coordFilename,"THERMOSTAT VELOCITY(IES)",error_msg);
          throw ProblemSetupException(error_msg, __FILE__, __LINE__);
        }
        if (lucretiusParse::skipComments(coordFile,buffer)) // Thermostat accel.
        {
          std::vector<std::string> parseTokens;
          Parse::tokenize(buffer, parseTokens);
          size_t maxTokens = parseTokens.size();
          if (maxTokens < 1) // Should be at least one acceleration
          {
            std::stringstream errorOut;
            errorOut << "ERROR:  No thermostat acceleration variables found in expected line in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Thermostat Acceleration Line:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          else if (maxTokens != thermostatX.size())
          {
            std::stringstream errorOut;
            errorOut << "ERROR:  Differing number of thermostat position and acceleration variables found in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Thermostat Positions parsed: " << thermostatX.size() << std::endl
                     << "  Thermostat Accelerations parsed: " << maxTokens << std::endl
                     << "  Acceleration input buffer:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);

          }
          else // Store thermostat accelerations
          {
            for (size_t Index=0; Index < maxTokens; ++Index) {
              thermostatA.push_back(Parse::stringToDouble(parseTokens[Index])
                                    * accelerationConversion);
            }
          }
        }
        else // Thermostat acceleration not found
        {
          lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"THERMOSTAT ACCELERATION(S)",error_msg);
          throw ProblemSetupException(error_msg, __FILE__, __LINE__);
        }
// Barostat extended lagrangian variables
        if (lucretiusParse::skipComments(coordFile,buffer)) // Barostat coords.
        {
          std::vector<std::string> parseTokens;
          Parse::tokenize(buffer, parseTokens);
          size_t maxTokens = parseTokens.size();
          if (maxTokens != 3) // Should be 3 barostat coordinates
          {
            std::stringstream errorOut;
            errorOut << "ERROR:  Barostat position should have three variables in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Input buffer:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          else  // Store barostat coords
          {
            barostatX = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                                       Parse::stringToDouble(parseTokens[1]),
                                       Parse::stringToDouble(parseTokens[2]));
            barostatX *= distanceConversion;
          }
        }
        else // No barostat coord section
        {
          lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"BAROSTAT COORDINATE(S)",error_msg);
          throw ProblemSetupException(error_msg, __FILE__, __LINE__);
        }
        if (lucretiusParse::skipComments(coordFile,buffer)) // Read barostat veloc.
        {
          std::vector<std::string> parseTokens;
          Parse::tokenize(buffer, parseTokens);
          size_t maxTokens = parseTokens.size();
          if (maxTokens != 3) // Should be 3 barostat velocities
          {
            std::stringstream errorOut;
            errorOut << "ERROR:  Barostat velocity should have three variables in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Input buffer:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          else // Store barostat velocities
          {
            barostatV = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                                       Parse::stringToDouble(parseTokens[1]),
                                       Parse::stringToDouble(parseTokens[2]));
            barostatV *= velocityConversion;
          }
        }
        else // No barostat velocity section
        {
          lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"BAROSTAT VELOCITY(IES)",error_msg);
          throw ProblemSetupException(error_msg, __FILE__, __LINE__);
        }
        if (lucretiusParse::skipComments(coordFile,buffer)) // Read barostat accel.
        {
          std::vector<std::string> parseTokens;
          Parse::tokenize(buffer, parseTokens);
          size_t maxTokens = parseTokens.size();
          if (maxTokens != 3) // Should be 3 barostat accelerations
          {
            std::stringstream errorOut;
            errorOut << "ERROR:  Barostat acceleration should have three variables in Lucretius coordinate file." << std::endl
                     << "  Filename:  " << coordFilename << std::endl
                     << "  Input buffer:  \"" << buffer << "\"" << std::endl;
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          else // Store barostat accelerations
          {
            barostatA = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                                       Parse::stringToDouble(parseTokens[1]),
                                       Parse::stringToDouble(parseTokens[2]));
            barostatA *= accelerationConversion;
          }
        }
        else // No barostat acceleration section
        {
          lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"BAROSTAT ACCELERATION(S)",error_msg);
          throw ProblemSetupException(error_msg, __FILE__, __LINE__);
        }
        coordFile.close();
      }
      else  // Found no coordinates in coordinate file
      {
        lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"COORDINATE SECTION",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
    else // Should never get here if fractory dispatch works correctly!
    {
      std::stringstream errorOut;
      errorOut << "ERROR:  Somehow parsing a forcefield of type " << ffType
               << " inside the routine intended for Lucretius forcefield parsing." << std::endl;
      throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
    }
  }
  else // No forcefield block
  {
    std::stringstream errorOut;
    errorOut << "ERROR:  Could not find the Forcefield block of the input file!" << std::endl;
    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
  }
  this->outputStatistics();
}

lucretiusAtomMap::~lucretiusAtomMap() {
  lucretiusMapIterator lMapIt;
  for(lMapIt = atomSet.begin(); lMapIt != atomSet.end(); ++lMapIt) {
    std::vector<atomData*>* currentArray = lMapIt->second;
    size_t numData = currentArray->size();
    for (size_t arrayIndex = 0; arrayIndex < numData; ++arrayIndex) {
      if ((*currentArray)[arrayIndex]) delete (*currentArray)[arrayIndex];
    }
    delete currentArray;
  }
}

void lucretiusAtomMap::outputStatistics() const {
  // Quick parse pass to see what we ended up with.
  size_t numAtomTypes=0;
  numAtomTypes = this->getNumberAtomTypes();
  proc0cout << "Constructed a Lucretius atom map with " << numAtomTypes << " atom types reported." << std::endl;
  size_t totalAtomCount = 0;
  std::vector<std::string> typeLabel;
  std::vector<size_t> numPerType;
  lucretiusMap::const_iterator it;
  for (it = atomSet.begin(); it != atomSet.end(); ++it) {
    totalAtomCount += it->second->size();
    numPerType.push_back(it->second->size());
    typeLabel.push_back(it->first);
  }
  if (numPerType.size() != typeLabel.size()) {
    proc0cerr << "Something's wrong here.  We have a mismatched number of sizes and labels!" << std::endl;
  }
  else {
    for (size_t idx=0; idx < typeLabel.size(); ++idx) {
      proc0cout << "Stored " << std::setw(5) << std::right << numPerType[idx] << " atoms with the label: \"" << typeLabel[idx] << "\"" << std::endl;
    }
    proc0cout << "Total atoms added: " << totalAtomCount << "\n" << std::endl;
  }

}

