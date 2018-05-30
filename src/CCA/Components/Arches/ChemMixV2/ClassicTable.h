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

    virtual  ~Interp_class() {
            if (d_allIndepVarNo.size()>1){
              delete &ind_1; 
            }
            delete &d_allIndepVarNo; 
            delete &indep; 
#ifdef UINTAH_ENABLE_KOKKOS
        // no delete needed due to kokkos smart pointers
#else
            delete &table2; 
#endif
                }

    virtual inline std::vector<double> find_val( const std::vector<double>& iv, const std::vector<int>& var_index) {return std::vector<double>(1,0.0);}

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

  protected:

    tableContainer  table2;  // All dependent variables
    const std::vector<int>&  d_allIndepVarNo; // size of independent variable array, for all independent variables
    const std::vector< std::vector <double> >&  indep;  // independent variables 1 to N-1
    const std::vector< std::vector <double > >&  ind_1; // independent variable N
  public:   // avoids re-order warning
    const ClassicTableInfo tableInfo; // variable names, units, and table keys

  };

  class Interp1 : public Interp_class {

  public:

    Interp1( const std::vector<int>& indepVarNo, tableContainer  table,
             const std::vector< std::vector <double> >& i1,  const ClassicTableInfo &cti)
      : Interp_class(table, indepVarNo, i1, i1, cti ) {
    }


    inline std::vector<double> find_val( const std::vector <double>& iv, const std::vector<int>& var_index) {

      std::vector<double> table_vals = std::vector<double>(2);
      std::vector<int> lo_index = std::vector<int>(1);
      std::vector<int> hi_index = std::vector<int>(1);
      int i1dep_ind = 0;
      int mid = 0;
      int lo_ind = 0;
      double iv_val = iv[0];
      double var_val = 0.0;
      std::vector<double> var_values (var_index.size(), 0.0 );

      //d_interpLock.lock();
      {

        int hi_ind = d_allIndepVarNo[0] - 1;

        if (ind_1[i1dep_ind][lo_ind] != iv_val && ind_1[i1dep_ind][hi_ind] != iv_val) {
          while ((hi_ind-lo_ind) > 1) {
            mid = (lo_ind+hi_ind)/2;
            if (ind_1[i1dep_ind][mid] > iv_val ) {
              hi_ind = mid;
            } else if (ind_1[i1dep_ind][mid] < iv_val) {
              lo_ind = mid;
            } else {
              // if (i1[i1dep_ind][mid] == iv[0])
              lo_ind = mid;
              hi_ind = mid;
            }
          }
        } else if (ind_1[i1dep_ind][lo_ind] == iv_val) {
          hi_ind = 1;
        } else {
          lo_ind = hi_ind-1;
        }

        lo_index[0] = lo_ind;
        hi_index[0] = hi_ind;

        if (iv_val < ind_1[i1dep_ind][0]) {
          hi_index[0] = 0;
          lo_index[0] = 0;
        }

        for (unsigned int i = 0; i < var_index.size(); i++) {

#ifdef UINTAH_ENABLE_KOKKOS
          table_vals[0] = table2(var_index[i], lo_index[0]);
          table_vals[1] = table2(var_index[i], hi_index[0]);
#else
          table_vals[0] = table2[var_index[i]][lo_index[0]];
          table_vals[1] = table2[var_index[i]][hi_index[0]];
#endif

          var_val = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[0][lo_index[0]])*(iv[0]-ind_1[0][lo_index[0]])+ table_vals[0];
          var_values[i] = var_val;
        }

      }
      //d_interpLock.unlock();

      return var_values;

    };
  };

  class Interp2 : public Interp_class {

  public:

    Interp2( const std::vector<int>& indepVarNo, tableContainer table,
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1, const ClassicTableInfo &cti )
      : Interp_class( table, indepVarNo, indep_headers, i1, cti ){}


    inline std::vector<double> find_val( const std::vector<double>& iv, const std::vector<int>& var_index) {

      std::vector<double> table_vals = std::vector<double>(4);
      std::vector<int> lo_index = std::vector<int>(2);
      std::vector<int> hi_index = std::vector<int>(2);
      int mid = 0;
      int lo_ind;
      int hi_ind;
      double iv_val;
      double var_val = 0.0;
      std::vector<double> var_values (var_index.size(), 0.0 );

      //d_interpLock.lock();
      {

        //binary search loop 2-> N
        for (int i = 1; i < 2; i++) {
          lo_ind = 0;
          hi_ind = d_allIndepVarNo[i] - 1;
          iv_val = iv[i];

          if (indep[i-1][lo_ind] != iv_val &&  indep[i-1][hi_ind] != iv_val) {
            while ((hi_ind-lo_ind) > 1) {
              mid = (lo_ind+hi_ind)/2;
              if (indep[i-1][mid] > iv_val ) {
                hi_ind = mid;
              } else if (indep[i-1][mid] < iv_val ){
                lo_ind = mid;
              } else {
                //if (indep_headers[i-1][mid] ==  iv[i])
                lo_ind = mid;
                hi_ind = mid;
              }
            }
          } else if (indep[i-1][lo_ind] == iv_val) {
            hi_ind = 1;
          } else {
            lo_ind = hi_ind-1;
          }
          lo_index[i] = lo_ind;
          hi_index[i] = hi_ind;

          if (iv_val < indep[i-1][0]) {
            lo_index[i] = 0;
            hi_index[i] = 0;
          }

        }

        //binary search for i1
        int i1dep_ind = lo_index[1];     //assume i1 is dep on last var

        lo_ind = 0;
        hi_ind = d_allIndepVarNo[0] - 1;
        iv_val = iv[0];

        if (ind_1[i1dep_ind][lo_ind] != iv_val && ind_1[i1dep_ind][hi_ind] != iv_val) {
          while ((hi_ind-lo_ind) > 1) {
            mid = (lo_ind+hi_ind)/2;
            if (ind_1[i1dep_ind][mid] > iv_val ) {
              hi_ind = mid;
            } else if (ind_1[i1dep_ind][mid] < iv_val) {
              lo_ind = mid;
            } else {
              //if (i1[i1dep_ind][mid] == iv[0])
              lo_ind = mid;
              hi_ind = mid;
            }
          }
        } else if (ind_1[i1dep_ind][lo_ind] == iv_val) {
          hi_ind = 1;
        } else {
          lo_ind = hi_ind-1;
        }

        lo_index[0] = lo_ind;
        hi_index[0] = hi_ind;

        if (iv_val < ind_1[i1dep_ind][0]) {
          hi_index[0] = 0;
          lo_index[0] = 0;
        }

        for (unsigned int i = 0; i < var_index.size(); i++) {
#ifdef UINTAH_ENABLE_KOKKOS

          table_vals[0] = table2(var_index[i], d_allIndepVarNo[0] * lo_index[1] + lo_index[0]);
          table_vals[1] = table2(var_index[i], d_allIndepVarNo[0] * lo_index[1] + hi_index[0]);
          table_vals[2] = table2(var_index[i], d_allIndepVarNo[0] * hi_index[1] + lo_index[0]);
          table_vals[3] = table2(var_index[i], d_allIndepVarNo[0] * hi_index[1] + hi_index[0]);
#else
          table_vals[0] = table2[var_index[i]][d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
          table_vals[1] = table2[var_index[i]][d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
          table_vals[2] = table2[var_index[i]][d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
          table_vals[3] = table2[var_index[i]][d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
#endif



          table_vals[0] = (table_vals[2] - table_vals[0])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[0];
          table_vals[1] = (table_vals[3] - table_vals[1])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[1];

          var_val = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+ table_vals[0];
          var_values[i] = var_val;
        }

      }
      //d_interpLock.unlock();

      return var_values;

    };
  };

  class Interp3 : public Interp_class {

  public:

    Interp3( const std::vector<int>& indepVarNo, tableContainer  table,
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1,  const ClassicTableInfo &cti)
      : Interp_class( table, indepVarNo, indep_headers, i1 ,cti) {}


    inline std::vector<double> find_val( const std::vector<double>& iv, const std::vector<int>& var_index) {

      std::vector<double> table_vals = std::vector<double>(8,0.0);
      std::vector<double> dist_vals = std::vector<double>(4,0.0); // make sure the default is zero
      std::vector<int> lo_index = std::vector<int>(4,0);
      std::vector<int> hi_index = std::vector<int>(4,0);
      int mid = 0;
      double var_val = 0.0;
      int lo_ind;
      int hi_ind;
      double iv_val;
      std::vector<double> var_values (var_index.size(), 0.0 );

      //d_interpLock.lock();
      {
        // binary search loop 2-> N
        for (int i = 1; i < 3; i++) {
          lo_ind = 0;
          hi_ind = d_allIndepVarNo[i] - 1;
          iv_val = iv[i];

          if (indep[i-1][lo_ind] != iv_val &&  indep[i-1][hi_ind] != iv_val) {
            while ((hi_ind-lo_ind) > 1) {
              mid = (lo_ind+hi_ind)/2;
              if (indep[i-1][mid] > iv_val ) {
                hi_ind = mid;
              } else if (indep[i-1][mid] < iv_val ){
                lo_ind = mid;
              } else {
                // if (indep_headers[i-1][mid] ==  iv[i])
                lo_ind = mid;
                hi_ind = mid;
              }
            }
          } else if (indep[i-1][lo_ind] == iv_val) {
            hi_ind = 1;
          } else {
            lo_ind = hi_ind-1;
          }
          lo_index[i+1] = lo_ind;
          hi_index[i+1] = hi_ind;

          if (iv_val < indep[i-1][0]) {
            lo_index[i+1] = 0;
            hi_index[i+1] = 0;
          }
        }

        int i1dep_ind1 = 0;
        // binary search for i1 low
        i1dep_ind1 = lo_index[3];
        lo_ind = 0;
        hi_ind = d_allIndepVarNo[0] - 1;
        iv_val = iv[0];
        if (ind_1[i1dep_ind1][lo_ind] != iv_val && ind_1[i1dep_ind1][hi_ind] != iv_val) {
          while ((hi_ind-lo_ind) > 1) {
            mid = (lo_ind+hi_ind)/2;
            if (ind_1[i1dep_ind1][mid] > iv_val ) {
              hi_ind = mid;
            } else if (ind_1[i1dep_ind1][mid] < iv_val) {
              lo_ind = mid;
            } else {
              // if (i1[i1dep_ind][mid] == iv[0])
              lo_ind = mid;
              hi_ind = mid;
            }
          }
        } else if (ind_1[i1dep_ind1][lo_ind] == iv_val) {
          hi_ind = 1;
        } else {
          lo_ind = hi_ind-1;
        }
        lo_index[0] = lo_ind;
        hi_index[0] = hi_ind;
        if (iv_val < ind_1[i1dep_ind1][0]) {
          hi_index[0] = 0;
          lo_index[0] = 0;
        }

        // binary search for i1 high
        int i1dep_ind2 = 0;
        i1dep_ind2 = hi_index[3];
        lo_ind = 0;
        hi_ind = d_allIndepVarNo[0] - 1;
        iv_val = iv[0];
        if (ind_1[i1dep_ind2][lo_ind] != iv_val && ind_1[i1dep_ind2][hi_ind] != iv_val) {
          while ((hi_ind-lo_ind) > 1) {
            mid = (lo_ind+hi_ind)/2;
            if (ind_1[i1dep_ind2][mid] > iv_val ) {
              hi_ind = mid;
            } else if (ind_1[i1dep_ind2][mid] < iv_val) {
              lo_ind = mid;
            } else {
              // if (i1[i1dep_ind][mid] == iv[0])
              lo_ind = mid;
              hi_ind = mid;
            }
          }
        } else if (ind_1[i1dep_ind2][lo_ind] == iv_val) {
          hi_ind = 1;
        } else {
          lo_ind = hi_ind-1;
        }
        lo_index[1] = lo_ind;
        hi_index[1] = hi_ind;
        if (iv_val < ind_1[i1dep_ind2][0]) {
          hi_index[1] = 0;
          lo_index[1] = 0;
        }

        for ( unsigned int i = 0; i < var_index.size(); i++ ) {

#ifdef UINTAH_ENABLE_KOKKOS
          table_vals[0] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * lo_index[2] + lo_index[0]);
          table_vals[1] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * lo_index[2] + hi_index[0]);
          table_vals[2] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * hi_index[2] + lo_index[0]);
          table_vals[3] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * hi_index[2] + hi_index[0]);
          table_vals[4] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * lo_index[2] + lo_index[1]);
          table_vals[5] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * lo_index[2] + hi_index[1]);
          table_vals[6] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * hi_index[2] + lo_index[1]);
          table_vals[7] = table2(var_index[i], d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * hi_index[2] + hi_index[1]);
#else
          table_vals[0] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * lo_index[2] + lo_index[0]];
          table_vals[1] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * lo_index[2] + hi_index[0]];
          table_vals[2] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * hi_index[2] + lo_index[0]];
          table_vals[3] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3] + d_allIndepVarNo[0] * hi_index[2] + hi_index[0]];
          table_vals[4] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * lo_index[2] + lo_index[1]];
          table_vals[5] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * lo_index[2] + hi_index[1]];
          table_vals[6] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * hi_index[2] + lo_index[1]];
          table_vals[7] = table2[var_index[i]][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3] + d_allIndepVarNo[0] * hi_index[2] + hi_index[1]];
#endif

          if (ind_1[i1dep_ind1][hi_index[0]]!=ind_1[i1dep_ind1][lo_index[0]]){
            dist_vals[0]=(iv[0]-ind_1[i1dep_ind1][lo_index[0]])/(ind_1[i1dep_ind1][hi_index[0]]-ind_1[i1dep_ind1][lo_index[0]]);
          }
          if (ind_1[i1dep_ind2][hi_index[1]]!=ind_1[i1dep_ind2][lo_index[1]]){
            dist_vals[1]=(iv[0]-ind_1[i1dep_ind2][lo_index[1]])/(ind_1[i1dep_ind2][hi_index[1]]-ind_1[i1dep_ind2][lo_index[1]]);
          }
          if (indep[0][hi_index[2]]!=indep[0][lo_index[2]]){
            dist_vals[2]=(iv[1]-indep[0][lo_index[2]])/(indep[0][hi_index[2]]-indep[0][lo_index[2]]);
          }
          if (indep[1][hi_index[3]]!=indep[1][lo_index[3]]){
            dist_vals[3]=(iv[2]-indep[1][lo_index[3]])/(indep[1][hi_index[3]]-indep[1][lo_index[3]]);
          }

          // First, we make the 2d plane in indep_1 space
          table_vals[0] = table_vals[0]*(1 - dist_vals[0]) + table_vals[1]*dist_vals[0];
          table_vals[1] = table_vals[4]*(1 - dist_vals[1]) + table_vals[5]*dist_vals[1];
          table_vals[2] = table_vals[2]*(1 - dist_vals[0]) + table_vals[3]*dist_vals[0];
          table_vals[3] = table_vals[6]*(1 - dist_vals[1]) + table_vals[7]*dist_vals[1];
          // Next, a line in indep_2 space
          table_vals[0] = table_vals[0]*(1 - dist_vals[2]) + table_vals[2]*dist_vals[2];
          table_vals[1] = table_vals[1]*(1 - dist_vals[2]) + table_vals[3]*dist_vals[2];
          // Finally, we interplate the new line in indep_3 space
          var_val = table_vals[0]*(1 - dist_vals[3]) + table_vals[1]*dist_vals[3];
          var_values[i] = var_val;

        }

      }
      //d_interpLock.unlock();
      return var_values;

    };
  };

  class Interp4 : public Interp_class {

  public:

    Interp4( const std::vector<int>& indepVarNo, tableContainer  table,
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1, const ClassicTableInfo &cti)
      : Interp_class(table, indepVarNo, indep_headers, i1, cti ){}

    inline std::vector<double> find_val( const std::vector<double>& iv, const std::vector<int>& var_index) {

      int mid = 0;
      double var_value = 0.0;
      int lo_ind;
      int hi_ind;
      double iv_val;
      std::vector<double> table_vals = std::vector<double>(16);
      std::vector<int> lo_index = std::vector<int>(4);
      std::vector<int> hi_index = std::vector<int>(4);
      std::vector<double> var_values (var_index.size(), 0.0 );

      //d_interpLock.lock();

      {

        // binary search loop 2-> N
        for (int i = 1; i < 4; i++) {
          lo_ind = 0;
          hi_ind = d_allIndepVarNo[i] - 1;
          iv_val = iv[i];

          if (indep[i-1][lo_ind] != iv_val &&  indep[i-1][hi_ind] != iv_val) {
            while ((hi_ind-lo_ind) > 1) {
              mid = (lo_ind+hi_ind)/2;
              if (indep[i-1][mid] > iv_val ) {
                hi_ind = mid;
              } else if (indep[i-1][mid] < iv_val ){
                lo_ind = mid;
              } else {
                //if (indep_headers[i-1][mid] ==  iv[i])
                lo_ind = mid;
                hi_ind = mid;
              }
            }
          } else if (indep[i-1][lo_ind] == iv_val) {
            hi_ind = 1;
          } else {
            lo_ind = hi_ind-1;
          }
          lo_index[i] = lo_ind;
          hi_index[i] = hi_ind;

          if (iv_val < indep[i-1][0]) {
            lo_index[i] = 0;
            hi_index[i] = 0;
          }
        }

        // binary search for i1
        int i1dep_ind = lo_index[3];     // assume i1 is dep on last var

        lo_ind = 0;
        hi_ind = d_allIndepVarNo[0] - 1;
        iv_val = iv[0];

        if (ind_1[i1dep_ind][lo_ind] != iv_val && ind_1[i1dep_ind][hi_ind] != iv_val) {
          while ((hi_ind-lo_ind) > 1) {
            mid = (lo_ind+hi_ind)/2;
            if (ind_1[i1dep_ind][mid] > iv_val ) {
              hi_ind = mid;
            } else if (ind_1[i1dep_ind][mid] < iv_val) {
              lo_ind = mid;
            } else {
              // if (i1[i1dep_ind][mid] == iv[0])
              lo_ind = mid;
              hi_ind = mid;
            }
          }
        } else if (ind_1[i1dep_ind][lo_ind] == iv_val) {
          hi_ind = 1;
        } else {
          lo_ind = hi_ind-1;
        }

        lo_index[0] = lo_ind;
        hi_index[0] = hi_ind;

        if (iv_val < ind_1[i1dep_ind][0]) {
          hi_index[0] = 0;
          lo_index[0] = 0;
        }

        for (unsigned int ii = 0; ii < var_index.size(); ii++) {

#ifdef UINTAH_ENABLE_KOKKOS
          table_vals[0] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]);
          table_vals[1] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]);
          table_vals[2] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]);
          table_vals[3] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]);
          table_vals[4] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]);
          table_vals[5] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]);
          table_vals[6] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]);
          table_vals[7] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]);
          table_vals[8] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]);
          table_vals[9] = table2 (var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]);
          table_vals[10] = table2(var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]);
          table_vals[11] = table2(var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]);
          table_vals[12] = table2(var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]);
          table_vals[13] = table2(var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]);
          table_vals[14] = table2(var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]);
          table_vals[15] = table2(var_index[ii], d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]);
#else
          table_vals[0] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
          table_vals[1] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
          table_vals[2] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
          table_vals[3] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
          table_vals[4] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
          table_vals[5] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
          table_vals[6] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
          table_vals[7] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
          table_vals[8] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
          table_vals[9] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
          table_vals[10] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
          table_vals[11] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
          table_vals[12] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
          table_vals[13] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
          table_vals[14] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
          table_vals[15] = table2[var_index[ii]][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
#endif
          // popvals

          int npts =0;
          for (int i = 3; i > 0; i--) {
            npts = (int)std::pow(2.0,i);
            for (int k=0; k < npts; k++) {
              table_vals[k] = (table_vals[k+npts]-table_vals[k])/(indep[i-1][lo_index[i]+1]-indep[i-1][lo_index[i]])*(iv[i]-indep[i-1][lo_index[i]])+table_vals[k];
            }
          }

          table_vals[0] = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[0];
          var_value = table_vals[0];
          var_values[ii] = var_value;
        }

      }
      //d_interpLock.unlock();

      return var_values;

    };
  };

  class InterpN : public Interp_class {

    public:

    InterpN( const std::vector<int>& indepVarNo, tableContainer  table,
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1, const ClassicTableInfo &cti)
      : Interp_class( table, indepVarNo, indep_headers, i1, cti ){
      int d_indepvarscount=indepVarNo.size();
      multiples = std::vector<int>(d_indepvarscount);
      multtemp = 0;
      for (int i = 0; i < d_indepvarscount; i++) {
        multtemp = 1;
        for (int j = 0; j<i; j++) {
          multtemp = multtemp * indepVarNo[j];
        }
        multiples[i] = multtemp;
      }

      int npts = (int)std::pow(2.0,d_indepvarscount);
      value_pop = std::vector< std::vector <bool> > (npts);

      for (int i =0; i < npts; i++) {
        value_pop[i] = std::vector<bool>(d_indepvarscount );
      }

      //bool matrix for use in lookup
      int temp_pts;
      double temp_pts_d;
      for (int i=0; i < npts; i++) {
        for (int j = d_indepvarscount-1; j >= 0; j--) {
          temp_pts_d = std::pow(2.0, j);
          temp_pts = (int)floor((i/temp_pts_d));
          if ((temp_pts % 2) == 0) {
            value_pop[i][j] = true;
          } else {
            value_pop[i][j] = false;
          }
        }
      }

      ivcount = d_indepvarscount;
      vals_size = npts;
    }

    inline std::vector<double> find_val( const std::vector<double>& iv, const std::vector<int>& var_index) {

      int mid = 0;
      double var_value = 0.0;
      int lo_ind;
      int hi_ind;
      double iv_val;
      std::vector<double> table_vals = std::vector<double>(vals_size);
      std::vector<int> lo_index = std::vector<int>(ivcount);
      std::vector<int> hi_index = std::vector<int>(ivcount);
      std::vector<double> var_values (var_index.size(), 0.0 );

      //d_interpLock.lock();

      {

        // binary search loop 2-> N
        for (int i = 1; i < ivcount; i++) {
          lo_ind = 0;
          hi_ind = d_allIndepVarNo[i] - 1;
          iv_val = iv[i];

          if (indep[i-1][lo_ind] != iv_val &&  indep[i-1][hi_ind] != iv_val) {
          while ((hi_ind-lo_ind) > 1) {
            mid = (lo_ind+hi_ind)/2;
            if (indep[i-1][mid] > iv_val ) {
              hi_ind = mid;
            } else if (indep[i-1][mid] < iv_val ){
              lo_ind = mid;
            }  else {
            // if (indep_headers[i-1][mid] ==  iv[i])
              lo_ind = mid;
              hi_ind = mid;
            }
          }
         } else if (indep[i-1][lo_ind] == iv_val) {
           hi_ind = 1;
         } else {
           lo_ind = hi_ind-1;
         }
          lo_index[i] = lo_ind;
          hi_index[i] = hi_ind;

          if (iv_val < indep[i-1][0]) {
            lo_index[i] = 0;
            hi_index[i] = 0;
          }
         }

         // binary search for i1
        int i1dep_ind = lo_index[ivcount-1];     // assume i1 is dep on last var
        lo_ind = 0;
        hi_ind = d_allIndepVarNo[0] - 1;
        iv_val = iv[0];

        if (ind_1[i1dep_ind][lo_ind] != iv_val && ind_1[i1dep_ind][hi_ind] != iv_val) {
          while ((hi_ind-lo_ind) > 1) {
            mid = (lo_ind+hi_ind)/2;
            if (ind_1[i1dep_ind][mid] > iv_val ) {
              hi_ind = mid;
            } else if (ind_1[i1dep_ind][mid] < iv_val) {
              lo_ind = mid;
            } else {
            // if (i1[i1dep_ind][mid] == iv[0])
              lo_ind = mid;
              hi_ind = mid;
            }
          }

        } else if (ind_1[i1dep_ind][lo_ind] == iv_val) {
          hi_ind = 1;
        } else {
          lo_ind = hi_ind-1;
        }

        lo_index[0] = lo_ind;
        hi_index[0] = hi_ind;

        if (iv_val < ind_1[i1dep_ind][0]) {
          hi_index[0] = 0;
          lo_index[0] = 0;
        }

        int npts = 0;

        npts = (int)std::pow(2.0,ivcount);
        int tab_index;

        for (unsigned int ii = 0; ii < var_index.size(); ii++) {
          // interpolant loop - 2parts read-in & calc
          for (int i=0; i < npts; i++) {
            tab_index = 0;
            for (int j = ivcount-1; j >= 0; j--) {
              if (value_pop[i][j]) { //determines hi/lo on bool
                tab_index = tab_index + multiples[j]*lo_index[j];
              } else {
                tab_index = tab_index + multiples[j]*hi_index[j];
              }
            }
#ifdef UINTAH_ENABLE_KOKKOS
            table_vals[i] = table2(var_index[ii], tab_index);
#else
            table_vals[i] = table2[var_index[ii]][tab_index];
#endif
          }

          for (int i = ivcount-1; i > 0; i--) {
            npts = (int)std::pow(2.0,i);
            for (int k=0; k < npts; k++) {
              table_vals[k] = (table_vals[k+npts]-table_vals[k])/(indep[i-1][lo_index[i]+1]-indep[i-1][lo_index[i]])*(iv[i]-indep[i-1][lo_index[i]])+table_vals[k];
            }
          }

          table_vals[0] = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[0];
          var_value = table_vals[0];
          var_values[ii] = var_value;

        }
      }
      //d_interpLock.unlock();

      return var_values;

    }

    protected:

    int ivcount;
    std::vector<int> multiples;
    std::vector <std::vector <bool> > value_pop;
    int multtemp;
    int vals_size;
  };
}
#endif
