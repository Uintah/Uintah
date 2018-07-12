/*
 * The MIT Licbense
 *
 * Copyright (c) 1997-2018 The University of Utah
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

//----- ClassicTable.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ClassicTable_h
#define Uintah_Component_Arches_ClassicTable_h


#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <sci_defs/kokkos_defs.h>



/**
 * @class  ClassicTable
 * @author Jeremy Thornock, Derek Harris (2017)
 * @date   Jan 2011
 *
 * @brief Table interface for those created with the Classic Arches Format
 *
 * @todo
 *
 * @details
 * This class stores object for storing dependent variables for a classic arches table.  
 * In the header there are also utilities for populating the table using an input file using the 
 * classic arches table format.
 * 
 *UPDATE: The table is now standalone, with static members for data construction.  THe table has the option to load
 *        only dependent variables requeseted by the class creating the object.  Additionally,  an access function
 *        that avoids indirection  within an i,j,k, loop is provided.
 *
*/


namespace Uintah {
#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
typedef Kokkos::View<double**,  Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> > tempTableContainer;
typedef Kokkos::View<const double**,   Kokkos::LayoutLeft,Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>  > tableContainer ;
#else
typedef std::vector<std::vector<double> > tempTableContainer;
typedef const std::vector<std::vector<double> > &tableContainer ;
#endif

struct ClassicTableInfo {

 const std::vector<std::vector<double> > &indep_headers;
 const std::vector<int>    &d_allIndepVarNum;            ///< std::vector storing the grid size for the Independent variables
  std::vector<std::string> d_allIndepVarNames;     ///< Vector storing all independent variable names from table file
  std::vector<std::string> d_savedDep_var;         ///< std::vector storing  saved dependent variable names from the table file
  std::vector<std::string>   d_allDepVarUnits;     
  std::map<std::string, double > d_constants;                           ///< List of constants in table header



  ClassicTableInfo(  // pay for full copy for arguments 3->6 (4 total)
  const std::vector<std::vector<double> > &arg1,
  const std::vector<int>    &arg2,                  ///< std::vector storing the grid size for the Independent variables
  const std::vector<std::string> &arg3,             ///< Vector storing all independent variable names from table file
  const std::vector<std::string> &arg4,             ///< std::vector storing  saved dependent variable names from the table file
  const std::vector<std::string> &arg5,              ///< Units for the dependent variables
  const std::map<std::string, double > &arg6 ) :     ///< List of constants in table header
  indep_headers(arg1),
  d_allIndepVarNum(arg2),      
  d_allIndepVarNames (arg3),   
  d_savedDep_var (arg4),     
  d_allDepVarUnits(arg5),    
  d_constants(arg6)  {}         

};

  /*********interp derived classes*****************************************/
  /** @brief A base class for Interpolation */
  class Interp_class {

  public:

    Interp_class( tableContainer  table,
                  const std::vector<int>& IndepVarNo,
                  const std::vector<std::vector<double> > & indepin,
                  const std::vector<std::vector<double> >& ind_1in,
                  const ClassicTableInfo &cti )
      : table2(table), d_allIndepVarNo(IndepVarNo), indep(indepin), ind_1(ind_1in), tableInfo(cti)
    {}

           ~Interp_class() {
            delete &ind_1; 
            delete &d_allIndepVarNo; 
            delete &indep; 
#ifdef UINTAH_ENABLE_KOKKOS
        // no delete needed due to kokkos smart pointers
#else
            delete &table2; 
#endif
                }



  template<class TYPE>
  void getState(std::vector< TYPE > &indep_storage,        
              std::vector<CCVariable<double> > &dep_storage,  std::vector<std::string> requestedInd_var,
              const Patch* patch, const std::vector<int> depVar_indices={} ){

    std::vector<int>     index_map (indep_storage.size());
    for (unsigned int ix = 0 ; ix<indep_storage.size(); ix++){
      index_map[ix]=ix;
    }

    //assume all variables are being read in from table data structure ( i would rather prune the dat structure then use a map)
    std::vector<int> depVarIndexes;
    if (depVar_indices.size()>0){
      depVarIndexes=depVar_indices;  // use user supplied dep var Index if supplied
    }else{
      depVarIndexes=std::vector<int> (dep_storage.size());
      for (unsigned int ix = 0 ; ix<dep_storage.size(); ix++) {
        depVarIndexes[ix]=ix;
      }
    }

    // Go through the patch and populate the requested state variables
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for(range,  [&]( int i,  int j, int k){

        std::vector<double> one_cell_iv1(indep_storage.size());
        // fill independent variables
        for (unsigned int ix = 0 ; ix<indep_storage.size(); ix++) {
        one_cell_iv1[ix]=indep_storage[index_map[ix]](i,j,k);
        }

        std::vector<double> depVarValues;
        //get all the needed varaible values from table with only one search
        depVarValues = find_val(one_cell_iv1, depVarIndexes );// would it be faster to pass by reference?  ~12 dependent variables for coal in 1-2017

        for (unsigned int ix = 0 ; ix<dep_storage.size(); ix++) {
        dep_storage[ix](i,j,k) = depVarValues[ix];
        }



        });
    }


    enum HighLow { iLow, iHigh};


    inline std::vector<double> find_val( const std::vector<double>& iv, const std::vector<int>& var_index) {
     //////////////---------------------table constants ------------------------------//////
     //////////////-------these parameters are constant for all I J K ----------------//////
     //////////////-----------------( make them class members? )----------------------//////
      const int nDim = d_allIndepVarNo.size();   // Number of dimensions
      const int npts = std::exp2(nDim); // double to int (danerous?)?
      const int oneD_switch= nDim == 1 ? 1 : 2;       // if/then switch for switching between 1-D and n-D tables
      const int nDim_withSwitch=nDim+1-oneD_switch;   //  nDim-1 except for 1D
      int index[2][nDim_withSwitch]; // high and low indexes
      int dliniate[nDim]; // This should only be once instead of for each I J K (Conversion factors for each dimension, for transforming from N-D to 1D)
     //////////////-------------------------------------------------------------------//////



      double table_vals[npts];   // container for values read from the table needed to compute the interpolent
      double distal_val[nDim-1+oneD_switch]; //  delta_x / DX   (nDim + 1  except for 1D)
      int table_indices[npts];   // container for table indices 
      int theSpecial[2][oneD_switch]; // high and low indexes of the special IV (first independent variable)
      


      std::vector<double> var_values (var_index.size(), 0.0 ); // This needs to be removed for portability and replaced with a view;

      dliniate[0]=1; 
      for( int  i=1 ; i<nDim; i++){
        dliniate[i]=dliniate[i-1]*d_allIndepVarNo[i-1]; // compute effective 1-D index
      }

      index[0][iLow]=0; // initialized for 1-D case
      index[0][iHigh]=0; // initialized for for 1-D case

       // ----------------perform search ------------//
       //  LINEAR search
      for (int j=0;  j< nDim-1 ; j++){
        if (iv[j+1] <  indep[j][d_allIndepVarNo[j+1]-1]){
          int i=1;
          while ( iv[j+1]  >indep[j][i]  ){
            i++;
          }
            index[iHigh][j]=i;
            index[iLow][j]=i-1;
            distal_val[j+2]=(iv[j+1]-indep[j][i-1])/(indep[j][i]-indep[j][i-1]);
        }else{
            index[iHigh][j]=d_allIndepVarNo[j+1]-1;
            index[iLow][j]=d_allIndepVarNo[j+1]-2;
            distal_val[j+2]=(iv[j+1]-indep[j][d_allIndepVarNo[j+1]-2])/(indep[j][d_allIndepVarNo[j+1]-1]-indep[j][d_allIndepVarNo[j+1]-2]);
        }
      }


       // ----------------perform search AGAIN for special IV  (indepednent variable-------//
       // LINEAR search
      for (int iSp=0;  iSp< oneD_switch; iSp++){ 
          const int cur_index=index[iSp][nDim_withSwitch-1];
        if (iv[0] <  ind_1[cur_index][d_allIndepVarNo[0]-1]){
          int i=1;
          while ( iv[0]  >ind_1[cur_index][i]  ){
            i++;
          }
            theSpecial[iHigh][iSp]=i;
            theSpecial[iLow][iSp]=i-1;
            distal_val[iSp]=(iv[0]-ind_1[cur_index][i-1])/(ind_1[cur_index][i]-ind_1[cur_index][i-1]);
        }else{
            theSpecial[iHigh][iSp]=d_allIndepVarNo[0]-1;
            theSpecial[iLow] [iSp]=d_allIndepVarNo[0]-2;
            distal_val[iSp]=(iv[0]-ind_1[cur_index][d_allIndepVarNo[0]-2])/(ind_1[cur_index][d_allIndepVarNo[0]-1]-ind_1[cur_index][d_allIndepVarNo[0]-2]);
        }
      }


          // compute table indices 
        for (int j=0; j<npts/2; j++){
          int table_index=0;  
          int base2=npts/4;
          int high_or_low=false;
          for (int i=1; i<nDim; i++){
            high_or_low=j / base2 % 2;
            table_index+=dliniate[i]*index[high_or_low][i-1];
            base2/=2;
          }
          table_indices[j]=table_index+theSpecial[iLow][high_or_low];
          table_indices[npts/2+j]=table_index+theSpecial[iHigh][high_or_low];
        }

  
        for (unsigned int k = 0; k < var_index.size(); k++) {
 /////      get values from table
        for (int j=0; j<npts; j++){
#ifdef UINTAH_ENABLE_KOKKOS
            table_vals[j]=table2(var_index[k],table_indices[j]);
#else
            table_vals[j]=table2[var_index[k]][table_indices[j]];
#endif
          }


 /////     do special interpolation for the first IV 
            int remaining_points=npts/2;
              for (int i=0; i < remaining_points; i++) {    
                table_vals[i]=table_vals[i]*(1. - distal_val[i % 2]) + table_vals[i+remaining_points]*distal_val[i % 2];
              }

 /////     do interpolation for all other IVs 
            for (int j = 0; j < nDim-1; j++) { 
              remaining_points /= 2;
              const double distl=distal_val[j+2];
              for (int i=0; i < remaining_points; i++) {
                table_vals[i]=table_vals[i]*(1. - distl) + table_vals[i+remaining_points]*distl;
              }
            }
            var_values[k] =table_vals[0];
        } // end K

      return var_values;

    }



  protected:

    tableContainer  table2;  // All dependent variables
    const std::vector<int>&  d_allIndepVarNo; // size of independent variable array, for all independent variables
    const std::vector< std::vector <double> >&  indep;  // independent variables 1 to N-1
    const std::vector< std::vector <double > >&  ind_1; // independent variable N
  public:   // avoids re-order warning
    const ClassicTableInfo tableInfo; // variable names, units, and table keys

  };






}
#endif
