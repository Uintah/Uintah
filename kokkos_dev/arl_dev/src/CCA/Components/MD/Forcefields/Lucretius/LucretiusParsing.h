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

/*
 * LucretiusParsing.h
 *
 *  Created on: Feb 3, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUSPARSING_H_
#define LUCRETIUSPARSING_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace lucretiusParse {

  inline bool skipComments(std::ifstream& fileHandle, std::string& buffer) {

    if (!fileHandle) return false; // file was EOF on entry

    std::getline(fileHandle,buffer); // Prime the buffer
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

  inline void generateUnexpectedEOFString(const std::string& filename,
                                   const std::string& addendum,
                                   std::string& buffer) {
    std::stringstream errorBuffer;
    errorBuffer << "ERROR:  Unexpected end of file " << filename << std::endl
                << "   -- Unable to locate the " << addendum << " section of the Lucretius forcefield file. ";
    buffer = errorBuffer.str();
  }

  inline void generateUnexpectedCoordinateEOF(const std::string& filename,
                                       const std::string& addendum,
                                       std::string& buffer) {
    std::stringstream errorBuffer;
    errorBuffer << "ERROR:  Unexpected end of file " << filename << std::endl
                << "   -- Unable to locate the " << addendum << " section of the Lucretius coordinate file. ";
    buffer = errorBuffer.str();

  }

  // Classes for reading connectivity.dat

    class bondConnectivityMap {

      public:
        bondConnectivityMap() {}
        ~bondConnectivityMap() {}
        friend std::istream& operator>>(std::istream&, bondConnectivityMap&);
      private:
        size_t firstAtom;
        size_t secondAtom;
        size_t bondPotentialIndex;
    };

    class bendConnectivityMap {
      public:
        bendConnectivityMap() {}
        ~bendConnectivityMap() {}
        friend std::istream& operator>>(std::istream&, bendConnectivityMap&);
      private:
        size_t firstAtom;
        size_t secondAtom;
        size_t thirdAtom;
        size_t bendPotentialIndex;
    };

    class torsionConnectivityMap {
      public:
        torsionConnectivityMap() {}
        ~torsionConnectivityMap() {}
        friend std::istream& operator>>(std::istream&, torsionConnectivityMap&);
      private:
        size_t firstAtom;
        size_t secondAtom;
        size_t thirdAtom;
        size_t fourthAtom;
        size_t torsionPotentialIndex;
    };

    class oopConnectivityMap {
      public:
        oopConnectivityMap() {}
        ~oopConnectivityMap() {}
        friend std::istream& operator>>(std::istream&, oopConnectivityMap&);
      private:
        size_t firstAtom;
        size_t secondAtom;
        size_t thirdAtom;
        size_t fourthAtom;
        size_t oopPotentialIndex;
        bool clockwise;
    };

    class chainConnectivityMap {
      public:
        chainConnectivityMap() {}
        ~chainConnectivityMap() {}
        friend std::istream& operator>>(std::istream&, chainConnectivityMap&);
      private:
        size_t numberAtoms;
        size_t numberBonds;
        size_t numberBends;
        size_t numberDihedrals;
        size_t numberOutOfPlane;
    };

    // Classes for reading ff.dat
    class nonbondedChargeMap {
      public:
        nonbondedChargeMap() {}
        ~nonbondedChargeMap() {}
        //friend std::istream& operator>>(std::istream&, NonbondedChargeType&);
      private:
        double Charge;
        double Polarizability;
        std::string Comment;
    };

    class nonbondedPotentialMap {
      public:
        nonbondedPotentialMap() {}
        ~nonbondedPotentialMap() {}
        friend std::istream& operator>>(std::istream&, nonbondedPotentialMap&);
      private:
        std::string Label;
        double A;
        double B;
        double C;
        double Mass;
        std::string PotentialTypeString;
        std::string Comment;
        std::vector<nonbondedChargeMap> ChargeTypes;
    };

    class nonbondedPotentialCrossTerm {
      public:
        nonbondedPotentialCrossTerm() {}
        nonbondedPotentialCrossTerm(std::string _T1, std::string _T2, double _A, double _B, double _C, std::string _comment)
            : firstTypeLabel(_T1), secondTypeLabel(_T2), A(_A), B(_B), C(_C), comment(_comment) {}
        ~nonbondedPotentialCrossTerm() {}
        friend std::istream& operator>>(std::istream&, nonbondedPotentialCrossTerm&);
        private:
        std::string firstTypeLabel;
        std::string secondTypeLabel;
        double A;
        double B;
        double C;
        std::string comment;
    };

    class bondPotentialMap {
      public:
        bondPotentialMap() {}
        bondPotentialMap(double _K, double _r0, double _rconstrained, std::string _comment)
            : harmonicConstant(_K), restDistance(_r0), constrainedDistance(_rconstrained), comment(_comment) {}
        ~bondPotentialMap() {}
        friend std::istream& operator>>(std::istream&, bondPotentialMap&);
        private:
        double harmonicConstant;
        double restDistance;
        double constrainedDistance;
        std::string comment;
    };

    class bendPotentialMap {
      public:
        bendPotentialMap() { }
        bendPotentialMap(double _K, double _theta, bool _Linear, std::string _comment)
            : harmonicConstant(_K), restAngle(_theta), linear(_Linear), comment(_comment) {}
        ~bendPotentialMap() {}
        friend std::istream& operator>>(std::istream&, bendPotentialMap&);
        private:
        double harmonicConstant;
        double restAngle;
        bool linear;
        std::string comment;
    };

    class torsionPotentialMap {
      public:
        torsionPotentialMap() {}
        torsionPotentialMap(size_t _numParams, std::vector<double>& _parameters, std::string _comment)
            : numberOfTerms(_numParams), parameters(_parameters), comment(_comment) {
          if (numberOfTerms != parameters.size()) {
            // Should throw an internal consistency error here, but for now I'm leaving it blank.  !JBH
            // probably should be:
            // throw LucretiusParseError();
          }
        }
        ~torsionPotentialMap() {}
        friend std::istream& operator>>(std::istream&, torsionPotentialMap&);
        private:
        size_t numberOfTerms;
        std::vector<double> parameters;
        std::string comment;
    };

    class oopPotentialMap {
      public:
        oopPotentialMap() {}
        oopPotentialMap(double _energeticConstant, std::string _comment);
        ~oopPotentialMap() {}
        friend std::istream& operator>>(std::istream&, oopPotentialMap&);
        private:
        double planarityConstant;
        std::string comment;
        bool constrainToPlanar;
    };

}


#endif /* LUCRETIUSPARSING_H_ */
