/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

 //----- Stream.h -----------------------------------------------

#ifndef Uintah_Components_Arches_Stream_h
#define Uintah_Components_Arches_Stream_h

/**************************************
CLASS
   Stream
   
   Class Stream creates and stores the mixing variables that are used in Arches

GENERAL INFORMATION
   Stream.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Revised by: Jennifer Spinti (spinti@crsim.utah.edu)
   
   Creation Date:   July 20, 2000
   Last Revised:   July 16, 2001
   
   C-SAFE 
   

KEYWORDS

DESCRIPTION

WARNING
   none

************************************************************************/
#include <vector>
#include <ostream>
#include <fstream>

namespace Uintah {
    // Low temperature limit; used in addStream
    const double TLIM = 200.0;
    const int NUM_DEP_VARS = 9; // includes all the vars in stateSpaceVector
                                //except vectors...
 
    class Stream {
    public:
      Stream();
      Stream(int numSpecies, int numElements);
      Stream(int numSpecies, int numElements, int numMixVars,
             int numRxnVars, bool lsoot);

      ///////////////////////////////////////////////////////////////////////
      //
      // Copy Constructor
      //         
      //
      Stream(const Stream& strm); // copy constructor

      // GROUP: Operators:
      ///////////////////////////////////////////////////////////////////////
      //
      // Assignment Operator 
      //         
      //
      Stream& operator=(const Stream &rhs);
      bool operator==(const Stream &rhs);
      ~Stream();
      Stream& linInterpolate(double upfactor, double lowfactor,
                             Stream& rightvalue);
      void print(std::ostream& out) const;
      void print_oneline(std::ofstream& out);
      //std::vector<double> convertStreamToVec(bool lsoot);
      std::vector<double> convertStreamToVec();
      void convertVecToStream(const std::vector<double>& vec_stateSpace, 
                              int numMixVars, int numRxnVars, bool lsoot);
      double getValue(int count);
      void normalizeStream();
      inline double getDensity() const {
        return d_density;
      }
      inline int getDepStateSpaceVars() const {
        // 4 correspond to density, pressure, temp, enthalpy
        return d_depStateSpaceVars;
      }
      inline double getEnthalpy() const {
        return d_enthalpy;
      }
      inline double getSensEnthalpy() const {
        return d_sensibleEnthalpy;
      }
      inline double getTemperature() const {
        return d_temperature;
      }
      inline double getPressure() const {
        return d_pressure;
      }
      inline bool getMoleBool() const {
        return d_mole;
      }
      inline double getCP() const {
        return d_cp;
      }
      inline double getCO2() const {
        if(d_speciesConcn.size()==0)
          return d_co2;
        else
          return d_speciesConcn[d_CO2index];
      }
      inline double getH2O() const {
        if(d_speciesConcn.size()==0)
          return d_h2o;
        else
          return d_speciesConcn[d_H2Oindex];
      }

      inline double getH2S() const {
        return d_h2s;
      }
      inline double getSO2() const {
        return d_so2;
      }
      inline double getSO3() const {
        return d_so3;
      }
      inline double getSULFUR() const {
        return d_sulfur;
      }

      inline double getS2()      const { return d_s2;   }
      inline double getSH()      const { return d_sh;   }
      inline double getSO()      const { return d_so;   }
      inline double getHSO2()    const { return d_hso2; }

      inline double getHOSO()    const { return d_hoso; }
      inline double getHOSO2()   const { return d_hoso2;}
      inline double getSN()      const { return d_sn;   }
      inline double getCS()      const { return d_cs;   }

      inline double getOCS()     const { return d_ocs;  }
      inline double getHSO()     const { return d_hso;  }
      inline double getHOS()     const { return d_hos;  }
      inline double getHSOH()    const { return d_hsoh; }

      inline double getH2SO()    const { return d_h2so; }
      inline double getHOSHO()   const { return d_hosho;}
      inline double getHS2()     const { return d_hs2;  }
      inline double getH2S2()    const { return d_h2s2; }

      inline double getCO2RATE() const { 
        return d_co2rate; 
      }
      inline double getSO2RATE() const { 
        return d_so2rate; 
      }

      inline double getCO() const {
        return d_co;
      }
      inline double getC2H2() const {
        return d_c2h2;
      }
      inline double getCH4() const {
        return d_ch4;
      }
      inline bool getSootBool() const {
        return d_lsoot;
      }
      inline double getSootFV() const {
        //return d_sootData[1];
        return d_sootFV;
      }
      inline double getfvtfive() const {
        return d_fvtfive;
      }
      inline double gettfour() const {
        return d_tfour;
      }
      inline double gettfive() const {
        return d_tfive;
      }
      inline double gettnine() const {
        return d_tnine;
      }
      inline double getqrg() const {
        return d_qrg;
      }
      inline double getqrs() const {
        return d_qrs;
      }
      inline double getRxnSource() const {
        return d_rxnVarRates[0];
      }
      inline double getdrhodf() const {
        return d_drhodf;
      }
      inline double getdrhodh() const {
        return d_drhodh;
      }
      inline double getheatLoss() const {
        return d_heatLoss;
      }
      inline double getMixMW() const {
        return d_mixmw; 
      }

    public:
      double d_pressure;          // Pa
      double d_density;           // kg/m^3
      double d_temperature;       // K
      double d_enthalpy;          // J/Kg
      double d_sensibleEnthalpy;  // J/Kg
      double d_moleWeight;
      double d_cp;                // J/Kg
      double d_drhodf;
      double d_drhodh;
      bool d_mole;
      int d_depStateSpaceVars;
      std::vector<double> d_speciesConcn; // Mass or mole fraction in
      // constructor; converted to mass fraction in addStream
      std::vector<double> d_atomNumbers;  // kg-atoms element I/kg mixture
      std::vector<double> d_rxnVarRates;  // mass fraction/s
      std::vector<double> d_rxnVarNorm;   // min/max values of rxn parameter
      std::vector<double> d_sootData;     // soot volume fraction and average diameter
      int d_numMixVars;
      int d_numRxnVars;
      bool d_lsoot;
      double d_sootFV;
      double d_co2;
      double d_h2o;

      double d_h2s;
      double d_so2;
      double d_so3;
      double d_sulfur;
      
      double d_s2;
      double d_sh;
      double d_so;
      double d_hso2;

      double d_hoso;
      double d_hoso2;
      double d_sn;
      double d_cs;

      double d_ocs;
      double d_hso;
      double d_hos;
      double d_hsoh;

      double d_h2so;
      double d_hosho;
      double d_hs2;
      double d_h2s2;


      double d_co;

      double d_c2h2;
      double d_ch4;
      double d_fvtfive;
      double d_tfour;
      double d_tfive;
      double d_tnine;
      double d_qrg;
      double d_qrs;

      double d_co2rate;
      double d_so2rate;
      double d_mixmw; 
      int d_CO2index; //Set to 0 in constructor.
                      //Value changed in ***MixingModel::computeProps
      int d_H2Oindex; //Set to 0 in constructor.
                      //Value changed in ***MixingModel::computeProps     
      double d_heatLoss;
    private:
      // includes all the vars except vectors...
      // increase the value if want to increase number of variables
      // WARNING: If you change this value, you must also change it in
      // all reaction model header files
      //

    }; // End class Stream

}  // End namespace Uintah

#endif

