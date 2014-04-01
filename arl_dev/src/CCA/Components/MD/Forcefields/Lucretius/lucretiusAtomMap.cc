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

#include <sstream>

using namespace Uintah;

lucretiusMapIterator lucretiusAtomMap::findValidAtomList(const std::string& searchLabel) {
  lucretiusMapIterator labelLocation = atomSet.find(searchLabel);

  if (labelLocation == atomSet.end()) { // Error:  Label not found
    std::stringstream errorOut;
    errorOut << "ERROR:  Attempt to access an atom type that does not exist." << std::endl
             << "  Forcefield Model --> Lucretius " << std::endl
             << "     Missing Label ---> \"" << searchLabel << "\"" << std::endl;
    throw InvalidValue(errorOut.str(), __FILE__, __LINE__);
    return (atomSet.end());
  }
  else return (labelLocation);
}

size_t lucretiusAtomMap::addAtomToList(const std::string& searchLabel, atomData* atomPtr) {
  lucretiusMapIterator labelLocation = findValidAtomList(searchLabel);
  if (labelLocation != atomSet.end()) { // Label already exists, so add to its entry.
    labelLocation->second->push_back(atomPtr);
    return (labelLocation->second->size());
  }
  else { // Label not found, so create a new entry for it.
    std::vector<atomData*>* newAtomDataArray = new std::vector<atomData*>;
    std::pair<lucretiusMapIterator,bool> result = atomSet.insert(lucretiusMapPair(searchLabel,newAtomDataArray));

    if (result.second) { // New vector successfully created
      // result.first is the iterator, so result.first->second is the vector pointer
      (result.first)->second->push_back(atomPtr);
      return(1);  // We just placed the first element in the array
    }
    else { // Could not create the new vector.  There's been some type of error somewhere.
      std::stringstream errorOut;
      errorOut << "ERROR:  Could not create new list for atom type." << std::endl
               << "  Forcefield Model --> Lucretius " << std::endl
               << "        Atom Label ---> \"" << searchLabel << "\"" << std::endl;
      throw InternalError(errorOut.str(), __FILE__, __LINE__);
      return(0);
    }
  }
}

lucretiusAtomMap::lucretiusAtomMap(const ProblemSpecP& spec, const SimulationStateP& shared_state) {

  ProblemSpecP coordSpec = spec->findBlock("MD")->findBlock("Forcefield");
  SCIRun::Vector box, barostatX, barostatV, barostatA;
  std::vector<double> thermostatX, thermostatV, thermostatA;

  if (coordSpec) { // Parse Forcefield block
    std::string ffType;
    coordSpec->getAttribute("type",ffType);

    if (ffType == "Lucretius") { // All as expected...
      std::string coordFilename;
      forcefieldType currentForcefieldType = Lucretius;
      coordSpec->get("coordinate_file", coordFilename);

      std::ifstream coordFile;
      coordFile.open(coordFilename.c_str(), std::ifstream::in);
      if (!coordFile) {
        throw ProblemSetupException("Could not open the Lucretius coordinate input file.", __FILE__, __LINE__);
      }

      std::string buffer;
      std::string error_msg;
      forcefieldType forcefield=Lucretius;
      if (lucretiusParse::skipComments(coordFile,buffer)) { // Skip comments
        bool coordinates = true;
        SCIRun::Point currentP;
        SCIRun::Vector currentV;
        std::string currentLabel;
        size_t atomCount=0;
        while (coordFile) { // Keep reading as long as we have data
          bool coordinates=true;
          while (buffer[0] != '*') {  // Parse coordinates
            // First parse the positions and type label
            getline(coordFile,buffer);
            if (coordinates) { // Current line is coordinates
              std::vector<std::string> parseTokens;
              Parse::tokenizeAtMost(buffer,parseTokens,3);
              if (parseTokens.size() != 4) { // Error:  Not a complete line!
                std::stringstream errorOut;
                errorOut << "ERROR:  Parsed incomplete line where coordinate line was expected." << std::endl
                         << "  Forcefield Type:  Lucretius" << std::endl
                         << "  Parsed Line:  \"" << buffer << "\"" << std::endl;
                throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
              }
              coordinates = false; // Next line is expected to be velocity
              double pX = Parse::stringToDouble(parseTokens[0]);
              double pY = Parse::stringToDouble(parseTokens[1]);
              double pZ = Parse::stringToDouble(parseTokens[2]);
              currentP = SCIRun::Point(pX,pY,pZ);
              std::ostringstream tempLabel(parseTokens[3].substr(0,3)); // Extract the text portion of the label.
              int chargeType = Parse::stringToInt(parseTokens[3].substr(3,std::string::npos));
              tempLabel << std::ios::left << chargeType;
              currentLabel = tempLabel.str();
            }
            else { // Current line is velocities
              double vX, vY, vZ;
              coordFile >> vX >> vY >> vZ;  // Velocities are far easier to parse
              currentV = SCIRun::Vector(vX,vY,vZ);
              ++atomCount; // Have coordinates, velocity, and label, and ID# now.  Create atom data.
               lucretiusAtomData* newAtom = new lucretiusAtomData(currentP, currentV, atomCount, currentLabel, forcefield);
              this->addAtomToList(currentLabel,newAtom);
              coordinates=true;
            }
          }  // End of coordinate section
          // Parse extended variables
          if (lucretiusParse::skipComments(coordFile,buffer)) { // Read box
            std::vector<std::string> parseTokens;
            Parse::tokenize(buffer, parseTokens);
            if (parseTokens.size() != 3) {
              std::stringstream errorOut;
              errorOut << "ERROR: Box size should contain three numbers in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Box Line:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
            }
            // Save the box
            box = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                                 Parse::stringToDouble(parseTokens[1]),
                                 Parse::stringToDouble(parseTokens[2]));
          } // Box info read
          else {  // Box info not found
            lucretiusParse::generateUnexpectedEOFString(coordFilename,"UNIT CELL DEFINITION",error_msg);
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          if (lucretiusParse::skipComments(coordFile,buffer)) { // Read thermostat position variables
            std::vector<std::string> parseTokens;
            Parse::tokenize(buffer, parseTokens);
            size_t maxTokens = parseTokens.size();
            if (maxTokens < 1) { // Should be at least one thermostat variable here
              std::stringstream errorOut;
              errorOut << "ERROR:  No thermostat position variables found in expected line in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Thermostat Position Line:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);
            }
            else { // Store thermostat positions
              for (size_t Index=0; Index < maxTokens; ++Index) {
                thermostatX.push_back(Parse::stringToDouble(parseTokens[Index]));
              }
            }
          }
          else {
            lucretiusParse::generateUnexpectedEOFString(coordFilename,"THERMOSTAT POSITION(S)",error_msg);
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          if (lucretiusParse::skipComments(coordFile,buffer)) { // Read thermostat velocity variables
            std::vector<std::string> parseTokens;
            Parse::tokenize(buffer, parseTokens);
            size_t maxTokens = parseTokens.size();
            if (maxTokens < 1) { // Should be at least one thermostat variable here
              std::stringstream errorOut;
              errorOut << "ERROR:  No thermostat velocity variables found in expected line in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Thermostat Velocity:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);
            }
            else if (maxTokens != thermostatX.size()) {
              std::stringstream errorOut;
              errorOut << "ERROR:  Differing number of thermostat position and velocity variables found in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Thermostat Positions parsed: " << thermostatX.size() << std::endl
                       << "  Thermostat Velocities parsed: " << maxTokens << std::endl
                       << "  Velocity input buffer:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);

            }
            else { // Store thermostat velocities
              for (size_t Index=0; Index < maxTokens; ++Index) {
                thermostatV.push_back(Parse::stringToDouble(parseTokens[Index]));
              }
            }
          }
          else {
            lucretiusParse::generateUnexpectedEOFString(coordFilename,"THERMOSTAT VELOCITY(IES)",error_msg);
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          if (lucretiusParse::skipComments(coordFile,buffer)) { // Read thermostat acceleration variables
            std::vector<std::string> parseTokens;
            Parse::tokenize(buffer, parseTokens);
            size_t maxTokens = parseTokens.size();
            if (maxTokens < 1) { // Should be at least one thermostat variable here
              std::stringstream errorOut;
              errorOut << "ERROR:  No thermostat acceleration variables found in expected line in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Thermostat Acceleration Line:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);
            }
            else if (maxTokens != thermostatX.size()) {
              std::stringstream errorOut;
              errorOut << "ERROR:  Differing number of thermostat position and acceleration variables found in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Thermostat Positions parsed: " << thermostatX.size() << std::endl
                       << "  Thermostat Accelerations parsed: " << maxTokens << std::endl
                       << "  Acceleration input buffer:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);

            }
            else { // Store thermostat accelerations
              for (size_t Index=0; Index < maxTokens; ++Index) {
                thermostatA.push_back(Parse::stringToDouble(parseTokens[Index]));
              }
            }
          }
          else {
            lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"THERMOSTAT ACCELERATION(S)",error_msg);
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          if (lucretiusParse::skipComments(coordFile,buffer)) { // Read barostat coordinate section
            std::vector<std::string> parseTokens;
            Parse::tokenize(buffer, parseTokens);
            size_t maxTokens = parseTokens.size();
            if (maxTokens != 3) { // Barostat coordinates should have three components
              std::stringstream errorOut;
              errorOut << "ERROR:  Barostat position should have three variables in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Input buffer:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);
            }
            else { // Store barostat coordinate
              barostatX = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                                         Parse::stringToDouble(parseTokens[1]),
                                         Parse::stringToDouble(parseTokens[2]));
            }
          }
          else { // No barostat coordinate section
            lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"BAROSTAT COORDINATE(S)",error_msg);
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          if (lucretiusParse::skipComments(coordFile,buffer)) { // Read barostat velocity section
            std::vector<std::string> parseTokens;
            Parse::tokenize(buffer, parseTokens);
            size_t maxTokens = parseTokens.size();
            if (maxTokens != 3) { // Should be three barostat velocity variables here
              std::stringstream errorOut;
              errorOut << "ERROR:  Barostat velocity should have three variables in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Input buffer:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);
            }
            else { // Store barostat velocity
              barostatV = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                                         Parse::stringToDouble(parseTokens[1]),
                                         Parse::stringToDouble(parseTokens[2]));
            }
          }
          else { // No barostat velocity section
            lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"BAROSTAT VELOCITY(IES)",error_msg);
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
          if (lucretiusParse::skipComments(coordFile,buffer)) { // Read barostat acceleration section
            std::vector<std::string> parseTokens;
            Parse::tokenize(buffer, parseTokens);
            size_t maxTokens = parseTokens.size();
            if (maxTokens != 3) { // Barostat acceleration vector has 3 components
              std::stringstream errorOut;
              errorOut << "ERROR:  Barostat acceleration should have three variables in Lucretius coordinate file." << std::endl
                       << "  Filename:  " << coordFilename << std::endl
                       << "  Input buffer:  \"" << buffer << "\"" << std::endl;
              throw ProblemSetupException(error_msg, __FILE__, __LINE__);
            }
            else { // Store barostat acceleration
              barostatA = SCIRun::Vector(Parse::stringToDouble(parseTokens[0]),
                                         Parse::stringToDouble(parseTokens[1]),
                                         Parse::stringToDouble(parseTokens[2]));
            }
          }
          else { // No barostat acceleration section
            lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"BAROSTAT ACCELERATION(S)",error_msg);
            throw ProblemSetupException(error_msg, __FILE__, __LINE__);
          }
        }
        coordFile.close();
      }
      else { // Found no coordinates in coordinate file
        lucretiusParse::generateUnexpectedCoordinateEOF(coordFilename,"COORDINATE SECTION",error_msg);
        throw ProblemSetupException(error_msg, __FILE__, __LINE__);
      }
    }
    else { // Should never get here if factory dispatch works correctly!
      std::stringstream errorOut;
      errorOut << "ERROR:  Somehow parsing a forcefield of type " << ffType
               << " inside the routine intended for Lucretius forcefield parsing." << std::endl;
      throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
    }
  }
  else { // No Forcefield block
    std::stringstream errorOut;
    errorOut << "ERROR:  Could not find the Forcefield block of the input file!" << std::endl;
    throw ProblemSetupException(errorOut.str(), __FILE__, __LINE__);
  }

}

