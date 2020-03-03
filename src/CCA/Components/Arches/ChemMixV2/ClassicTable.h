/*
 * The MIT Licbense
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#define MAX_TABLE_DIMENSION 3
#define MAX_TABLE_READS 8 // pow(2,max_table_dimension)

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
template <typename MemSpace>
using tempIntContainer   = Kokkos::View<int*,     Kokkos::LayoutLeft, MemSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> >;
template <typename MemSpace>
using intContainer       = Kokkos::View<int*,     Kokkos::LayoutLeft, MemSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> >;
template <typename MemSpace>
using tempTableContainer = Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> >;
template <typename MemSpace>
using tableContainer     = Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess> >;
#else
typedef std::vector<int> tempIntContainer;
typedef const std::vector<int> &intContainer;
typedef std::vector<std::vector<double> > tempTableContainer;
typedef const std::vector<std::vector<double> > &tableContainer ;
#endif

struct ClassicTableInfo {
#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
  tableContainer<Kokkos::HostSpace> indep_headers;
  intContainer<Kokkos::HostSpace> d_allIndepVarNum;            ///< std::vector storing the grid size for the Independent variables
#else
  const std::vector<std::vector<double> > &indep_headers;
  const std::vector<int> &d_allIndepVarNum;            ///< std::vector storing the grid size for the Independent variables
#endif
  std::vector<std::string> d_allIndepVarNames;     ///< Vector storing all independent variable names from table file
  std::vector<std::string> d_savedDep_var;         ///< std::vector storing  saved dependent variable names from the table file
  std::vector<std::string>   d_allDepVarUnits;
  std::map<std::string, double > d_constants;                           ///< List of constants in table header



  ClassicTableInfo(  // pay for full copy for arguments 3->6 (4 total)
#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
  tableContainer<Kokkos::HostSpace> arg1,
  intContainer<Kokkos::HostSpace> arg2,
#else
  const std::vector<std::vector<double> >&arg1,
  const std::vector<int> &arg2,
#endif
  const std::vector<std::string> &arg3,             ///< Vector storing all independent variable names from table file
  const std::vector<std::string> &arg4,             ///< std::vector storing  saved dependent variable names from the table file
  const std::vector<std::string> &arg5,              ///< Units for the dependent variables
  const std::map<std::string, double > &arg6
 ) :     ///< List of constants in table header
  indep_headers(arg1),
  d_allIndepVarNum(arg2),
  d_allIndepVarNames (arg3),
  d_savedDep_var (arg4),
  d_allDepVarUnits(arg5),
  d_constants(arg6)  {}

};

  /*********interp derived classes*****************************************/
  /** @brief A base class for Interpolation */
  template <unsigned int max_dep_var>
  class Interp_class {

  public:

#ifdef UINTAH_ENABLE_KOKKOS  // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
    Interp_class( tableContainer<Kokkos::HostSpace>  table,
                  intContainer<Kokkos::HostSpace> IndepVarNo,
                  tableContainer<Kokkos::HostSpace> indepin,
                  tableContainer<Kokkos::HostSpace> ind_1in,
#else
    Interp_class( tableContainer  table,
                  intContainer IndepVarNo,
                  tableContainer indepin,
                  tableContainer ind_1in,
#endif
                  const ClassicTableInfo &cti )
      : table2(table), d_allIndepVarNo(IndepVarNo), indep(indepin), ind_1(ind_1in), tableInfo(cti)
    {
#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
           int numDim=d_allIndepVarNo.size();
           int max_size=0;
           int size=d_allIndepVarNo(0); // size of a single dep variable
           for (int i = 0; i < numDim- 1; i++) {
            max_size=max(max_size, d_allIndepVarNo(i+1)); // pad this non-square portion of the table = (
            size*=d_allIndepVarNo(i+1);
           }

           g_d_allIndepVarNo= intContainer<Kokkos::CudaSpace>
("COPY_array_of_ind_var_sizes",numDim);            ///< std::vector storing the grid size for the Independent variables
           g_indep= tableContainer<Kokkos::CudaSpace>
("COPY_secondary_independent_variables",numDim-1,max_size);
           g_ind_1= tableContainer<Kokkos::CudaSpace>
("COPY_primary_independent_variable",d_allIndepVarNo(numDim-1),d_allIndepVarNo(0));
           g_table2= tableContainer<Kokkos::CudaSpace>
("COPY_ClassicMixingTable",table2.size()/size,size);


           Kokkos::deep_copy(g_d_allIndepVarNo,d_allIndepVarNo);
           Kokkos::deep_copy(g_indep,indep);
           Kokkos::deep_copy(g_ind_1,ind_1);
           Kokkos::deep_copy(g_table2,table2);
#endif
}

           ~Interp_class() {
#ifdef UINTAH_ENABLE_KOKKOS
        // no delete needed due to kokkos smart pointers
#else
            delete &ind_1;
            delete &d_allIndepVarNo;
            delete &indep;
            delete &table2;
#endif
             }

// WORKING!!!!!!!!!!!!!!!!!!!
  template <typename ExecSpace,typename MemSpace ,class TYPE_1 , class TYPE_2  >
  void getState(ExecutionObject<ExecSpace, MemSpace>& execObj, TYPE_1 &indep_storage,
              TYPE_2 &dep_storage,
              const Patch* patch, const struct1DArray<int, max_dep_var>  depVar_indices=struct1DArray<int,max_dep_var>(0) ){

    struct1DArray<int,MAX_TABLE_DIMENSION>     index_map (indep_storage.runTime_size);
    for (unsigned int ix = 0 ; ix<indep_storage.runTime_size; ix++){
      index_map[ix]=ix;
    }

    //assume all variables are being read in from table data structure ( i would rather prune the dat structure then use a map)
    struct1DArray<int,max_dep_var> var_index(depVar_indices.runTime_size);
    if (depVar_indices.runTime_size>0){
      //depVarIndexes=depVar_indices;  // use user supplied dep var Index if supplied
      for (unsigned int i=0; i<depVar_indices.runTime_size; i++){
        var_index[i]=depVar_indices[i];  // use user supplied dep var Index if supplied
      }
    }else{
      var_index=struct1DArray<int,max_dep_var> (dep_storage.runTime_size);
      for (unsigned int ix = 0 ; ix<dep_storage.runTime_size; ix++) {
        var_index[ix]=ix;
      }
    }
     // TDMS - > template defined memoryspace
      auto TDMS_d_allIndepVarNo=getInts<MemSpace>();
      auto TDMS_indep= getSecondaryVar<MemSpace>();
      auto TDMS_ind_1= getPrimaryVar<MemSpace>();
      auto TDMS_table2= getTable<MemSpace>();

      const int nDim = TDMS_d_allIndepVarNo.size();   // Number of dimensions
      const int npts = std::exp2(nDim); // double to int (danerous?)?
      const int oneD_switch= nDim == 1 ? 1 : 2;       // if/then switch for switching between 1-D and n-D tables
      const int nDim_withSwitch=nDim+1-oneD_switch;   //  nDim-1 except for 1D
    // Go through the patch and populate the requested state variables
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    Uintah::parallel_for<ExecSpace>(execObj,range, KOKKOS_LAMBDA ( int i,  int j, int k){

        struct1DArray<double,MAX_TABLE_DIMENSION> iv(indep_storage.runTime_size);
        // fill independent variables
        for (unsigned int ix = 0 ; ix<indep_storage.runTime_size; ix++) {
        iv[ix]=indep_storage[index_map[ix]](i,j,k);
        }

        struct1DArray<double,max_dep_var> var_values;

        //find_val_wrapper<MemSpace>(one_cell_iv1,depVarIndexes,depVarValues);
        //find_val<MemSpace>(one_cell_iv1,depVarIndexes,depVarValues, TDMS_table2, TDMS_d_allIndepVarNo, TDMS_indep, TDMS_ind_1);
        //find_val_type_correct<MemSpace>(one_cell_iv1,depVarIndexes,depVarValues, TDMS_table2, TDMS_d_allIndepVarNo, TDMS_indep, TDMS_ind_1);
        
       //COPY AND PASTE FIND_VAL DUE TO INTERNAL COMPILER ERROR?!!?!?  
     //////////////---------------------table constants ------------------------------//////
     //////////////-------these parameters are constant for all I J K ----------------//////
     //////////////-----------------( make them class members? )----------------------//////
      int dliniate[MAX_TABLE_DIMENSION]; // This should only be once instead of for each I J K (Conversion factors for each dimension, for transforming from N-D to 1D)
     //////////////-------------------------------------------------------------------//////

      dliniate[0]=1; 
      for( int  i=1 ; i<nDim; i++){
#ifdef UINTAH_ENABLE_KOKKOS
        dliniate[i]=dliniate[i-1]*TDMS_d_allIndepVarNo(i-1); // compute effective 1-D index
#else
        dliniate[i]=dliniate[i-1]*TDMS_d_allIndepVarNo[i-1]; // compute effective 1-D index
#endif
      }


      double table_vals[MAX_TABLE_READS];   // container for values read from the table needed to compute the interpolent
      double distal_val[MAX_TABLE_DIMENSION+1]; //  delta_x / DX   (nDim + 1  except for 1D)
      int table_indices[MAX_TABLE_READS];   // container for table indices 
      int index[2][MAX_TABLE_DIMENSION-1]; // high and low indexes
      int theSpecial[2][2]; // high and low indexes of the special IV (first independent variable)
     //////////////-------------------------------------------------------------------//////





      index[0][iLow]=0; // initialized for 1-D case
      index[0][iHigh]=0; // initialized for for 1-D case

       // ----------------perform search ------------//
       //  LINEAR search
      for (int j=0;  j< nDim-1 ; j++){
        
#ifdef UINTAH_ENABLE_KOKKOS
        if (iv[j+1] <  TDMS_indep(j, TDMS_d_allIndepVarNo(j+1)-1))
#else
        if (iv[j+1] <  TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-1])
#endif
         {
          int i=1;
#ifdef UINTAH_ENABLE_KOKKOS
          while ( iv[j+1]  >TDMS_indep(j, i)  )
#else
          while ( iv[j+1]  >TDMS_indep[j][i]  )
#endif
          {
            i++;
          }
            index[iHigh][j]=i;
            index[iLow][j]=i-1;
#ifdef UINTAH_ENABLE_KOKKOS
            distal_val[j+2]=(iv[j+1]-TDMS_indep(j, i-1))/(TDMS_indep(j, i)-TDMS_indep(j, i-1));
#else
            distal_val[j+2]=(iv[j+1]-TDMS_indep[j][i-1])/(TDMS_indep[j][i]-TDMS_indep[j][i-1]);
#endif
        }else{
#ifdef UINTAH_ENABLE_KOKKOS
            index[iHigh][j]=TDMS_d_allIndepVarNo(j+1)-1;
            index[iLow][j]=TDMS_d_allIndepVarNo(j+1)-2;
            distal_val[j+2]=(iv[j+1]-TDMS_indep(j, TDMS_d_allIndepVarNo[j+1]-2))/(TDMS_indep(j, TDMS_d_allIndepVarNo[j+1]-1)-TDMS_indep(j, TDMS_d_allIndepVarNo(j+1)-2));
#else
            index[iHigh][j]=TDMS_d_allIndepVarNo[j+1]-1;
            index[iLow][j]=TDMS_d_allIndepVarNo[j+1]-2;
            distal_val[j+2]=(iv[j+1]-TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-2])/(TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-1]-TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-2]);
#endif
        }
      }


       // ----------------perform search AGAIN for special IV  (indepednent variable-------//
       // LINEAR search
      for (int iSp=0;  iSp< oneD_switch; iSp++){
          const int cur_index=index[iSp][nDim_withSwitch-1];
#ifdef UINTAH_ENABLE_KOKKOS
        if (iv[0] <  TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo(0)-1))
#else
        if (iv[0] <  TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-1])
#endif
       {
          int i=1;
#ifdef UINTAH_ENABLE_KOKKOS
          while ( iv[0]  >TDMS_ind_1(cur_index, i)  )
#else
          while ( iv[0]  >TDMS_ind_1[cur_index][i]  )
#endif
          {
            i++;
          }
            theSpecial[iHigh][iSp]=i;
            theSpecial[iLow][iSp]=i-1;
#ifdef UINTAH_ENABLE_KOKKOS
            distal_val[iSp]=(iv[0]-TDMS_ind_1(cur_index, i-1))/(TDMS_ind_1(cur_index, i)-TDMS_ind_1(cur_index, i-1));
#else
            distal_val[iSp]=(iv[0]-TDMS_ind_1[cur_index][i-1])/(TDMS_ind_1[cur_index][i]-TDMS_ind_1[cur_index][i-1]);
#endif
        }else{
#ifdef UINTAH_ENABLE_KOKKOS
            theSpecial[iHigh][iSp]=TDMS_d_allIndepVarNo(0)-1;
            theSpecial[iLow] [iSp]=TDMS_d_allIndepVarNo(0)-2;
            distal_val[iSp]=(iv[0]-TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo(0)-2))/(TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo[0]-1)-TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo(0)-2));
#else
            theSpecial[iHigh][iSp]=TDMS_d_allIndepVarNo[0]-1;
            theSpecial[iLow] [iSp]=TDMS_d_allIndepVarNo[0]-2;
            distal_val[iSp]=(iv[0]-TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-2])/(TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-1]-TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-2]);
#endif
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


        for (unsigned int k = 0; k < var_index.runTime_size; k++) {
 /////      get values from table
        for (int j=0; j<npts; j++){
#ifdef UINTAH_ENABLE_KOKKOS
            table_vals[j]=TDMS_table2(var_index[k],table_indices[j]);
#else
            table_vals[j]=TDMS_table2[var_index[k]][table_indices[j]];
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

        for (unsigned int ix = 0 ; ix<dep_storage.runTime_size; ix++) {
        dep_storage[ix](i,j,k) = var_values[ix];
        }



        });
    }


// DUE TO USING THE PORTABILITY API INCORRECTLY ( couldn't figure it out without c+= 14)
#ifdef UINTAH_ENABLE_KOKKOS
    template< typename MemSpace, unsigned int numOfDep>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, void >::type
    find_val_type_correct( const struct1DArray<double,MAX_TABLE_DIMENSION>& one_cell_iv1, const struct1DArray<int,numOfDep>& depVarIndexes, struct1DArray<double,numOfDep>& depVarValues,
   tableContainer<MemSpace>  TDMS_table2,  
   intContainer<MemSpace>    TDMS_d_allIndepVarNo,
   tableContainer<MemSpace>  TDMS_indep,  
   tableContainer<MemSpace>  TDMS_ind_1 
    ) const {
       find_val<MemSpace>(one_cell_iv1,depVarIndexes,depVarValues,TDMS_table2,TDMS_d_allIndepVarNo,TDMS_indep,TDMS_ind_1);
    }
#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
    template< typename MemSpace, unsigned int numOfDep>
    KOKKOS_INLINE_FUNCTION 
    typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, void >::type
    find_val_type_correct( const struct1DArray<double,MAX_TABLE_DIMENSION>& one_cell_iv1, const struct1DArray<int,numOfDep>& depVarIndexes, struct1DArray<double,numOfDep>& depVarValues,
   tableContainer<MemSpace>  TDMS_table2,  
   intContainer<MemSpace>    TDMS_d_allIndepVarNo,
   tableContainer<MemSpace>  TDMS_indep,  
   tableContainer<MemSpace>  TDMS_ind_1 
    ) const {
      //printf("GPU table reading is being done incorrectly by the Arches Developers; Use CPU for this application.\n");
    find_val<MemSpace>(one_cell_iv1,depVarIndexes,depVarValues,TDMS_table2,TDMS_d_allIndepVarNo,TDMS_indep,TDMS_ind_1);
    }
#endif // end HAVE_CUDA && KOKKOS_ENABLE_CUDA
#endif // end UINTAH_ENABLE_KOKKOS

    template< typename MemSpace, unsigned int numOfDep>
#ifdef UINTAH_ENABLE_KOKKOS
    KOKKOS_INLINE_FUNCTION 
#else
   inline
#endif
    typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, void >::type
    find_val_type_correct( const struct1DArray<double,MAX_TABLE_DIMENSION>& one_cell_iv1, const struct1DArray<int,numOfDep>& depVarIndexes, struct1DArray<double,numOfDep>& depVarValues,
#ifdef UINTAH_ENABLE_KOKKOS
   tableContainer<Kokkos::HostSpace>  TDMS_table2,  
   intContainer<Kokkos::HostSpace>    TDMS_d_allIndepVarNo,
   tableContainer<Kokkos::HostSpace>  TDMS_indep,  
   tableContainer<Kokkos::HostSpace>  TDMS_ind_1 
#else
   tableContainer  TDMS_table2,  
   intContainer    TDMS_d_allIndepVarNo,
   tableContainer  TDMS_indep,  
   tableContainer  TDMS_ind_1 
#endif
 ) const { 
#ifdef UINTAH_ENABLE_KOKKOS // We have to do this because we don't want to store the table in kokkos::hostSpace AND uintahSpaces::HostSpace
      find_val<Kokkos::HostSpace>(one_cell_iv1,depVarIndexes,depVarValues,TDMS_table2,TDMS_d_allIndepVarNo,TDMS_indep,TDMS_ind_1);
#else
      find_val<UintahSpaces::HostSpace>(one_cell_iv1,depVarIndexes,depVarValues,TDMS_table2,TDMS_d_allIndepVarNo,TDMS_indep,TDMS_ind_1);
#endif
    }


#ifdef UINTAH_ENABLE_KOKKOS
    template< typename MemSpace, unsigned int numOfDep>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, void >::type
    find_val_wrapper( const struct1DArray<double,MAX_TABLE_DIMENSION>& one_cell_iv1, const struct1DArray<int,numOfDep>& depVarIndexes, struct1DArray<double,numOfDep>& depVarValues){
       find_val<MemSpace>(one_cell_iv1,depVarIndexes,depVarValues,table2,d_allIndepVarNo,indep,ind_1);
    }
#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
    template< typename MemSpace, unsigned int numOfDep>
    KOKKOS_INLINE_FUNCTION 
    typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, void >::type
    find_val_wrapper( const struct1DArray<double,MAX_TABLE_DIMENSION>& one_cell_iv1, const struct1DArray<int,numOfDep>& depVarIndexes, struct1DArray<double,numOfDep>& depVarValues){
      //printf("GPU table reading is being done incorrectly by the Arches Developers; Use CPU for this application.\n");
    find_val<MemSpace>(one_cell_iv1,depVarIndexes,depVarValues,g_table2,g_d_allIndepVarNo,g_indep,g_ind_1);
    }
#endif // end HAVE_CUDA && KOKKOS_ENABLE_CUDA
#endif // end UINTAH_ENABLE_KOKKOS

    template< typename MemSpace, unsigned int numOfDep>
#ifdef UINTAH_ENABLE_KOKKOS
    KOKKOS_INLINE_FUNCTION 
#else
   inline
#endif
    typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, void >::type
    find_val_wrapper( const struct1DArray<double,MAX_TABLE_DIMENSION>& one_cell_iv1, const struct1DArray<int,numOfDep>& depVarIndexes, struct1DArray<double,numOfDep>& depVarValues){
#ifdef UINTAH_ENABLE_KOKKOS // We have to do this because we don't want to store the table in kokkos::hostSpace AND uintahSpaces::HostSpace
      find_val<Kokkos::HostSpace>(one_cell_iv1,depVarIndexes,depVarValues,table2,d_allIndepVarNo,indep,ind_1);
#else
      find_val<UintahSpaces::HostSpace>(one_cell_iv1,depVarIndexes,depVarValues,table2,d_allIndepVarNo,indep,ind_1);
#endif
    }

    enum HighLow { iLow, iHigh};

    template< typename MemSpace, unsigned int numOfDep>
#ifdef UINTAH_ENABLE_KOKKOS
    KOKKOS_INLINE_FUNCTION 
#else
   inline
#endif
    void find_val( const struct1DArray<double,MAX_TABLE_DIMENSION>& iv, const struct1DArray<int,numOfDep>& var_index, struct1DArray<double,numOfDep>& var_values, 
#ifdef UINTAH_ENABLE_KOKKOS
   tableContainer<MemSpace>  TDMS_table2,  
   intContainer<MemSpace>    TDMS_d_allIndepVarNo,
   tableContainer<MemSpace>  TDMS_indep,  
   tableContainer<MemSpace>  TDMS_ind_1 
#else
   tableContainer  TDMS_table2,  
   intContainer    TDMS_d_allIndepVarNo,
   tableContainer  TDMS_indep,  
   tableContainer  TDMS_ind_1 
#endif
 ) const { 


     //////////////---------------------table constants ------------------------------//////
     //////////////-------these parameters are constant for all I J K ----------------//////
     //////////////-----------------( make them class members? )----------------------//////
      const int nDim = TDMS_d_allIndepVarNo.size();   // Number of dimensions
      const int npts = std::exp2(nDim); // double to int (danerous?)?
      const int oneD_switch= nDim == 1 ? 1 : 2;       // if/then switch for switching between 1-D and n-D tables
      const int nDim_withSwitch=nDim+1-oneD_switch;   //  nDim-1 except for 1D
      int dliniate[MAX_TABLE_DIMENSION]; // This should only be once instead of for each I J K (Conversion factors for each dimension, for transforming from N-D to 1D)
     //////////////-------------------------------------------------------------------//////



      double table_vals[MAX_TABLE_READS];   // container for values read from the table needed to compute the interpolent
      double distal_val[MAX_TABLE_DIMENSION+1]; //  delta_x / DX   (nDim + 1  except for 1D)
      int table_indices[MAX_TABLE_READS];   // container for table indices
      int index[2][MAX_TABLE_DIMENSION-1]; // high and low indexes
      int theSpecial[2][2]; // high and low indexes of the special IV (first independent variable)
     //////////////-------------------------------------------------------------------//////



      dliniate[0]=1;
      for( int  i=1 ; i<nDim; i++){
#ifdef UINTAH_ENABLE_KOKKOS
        dliniate[i]=dliniate[i-1]*TDMS_d_allIndepVarNo(i-1); // compute effective 1-D index
#else
        dliniate[i]=dliniate[i-1]*TDMS_d_allIndepVarNo[i-1]; // compute effective 1-D index
#endif
      }

      index[0][iLow]=0; // initialized for 1-D case
      index[0][iHigh]=0; // initialized for for 1-D case

       // ----------------perform search ------------//
       //  LINEAR search
      for (int j=0;  j< nDim-1 ; j++){
        
#ifdef UINTAH_ENABLE_KOKKOS
        if (iv[j+1] <  TDMS_indep(j, TDMS_d_allIndepVarNo(j+1)-1))
#else
        if (iv[j+1] <  TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-1])
#endif
         {
          int i=1;
#ifdef UINTAH_ENABLE_KOKKOS
          while ( iv[j+1]  >TDMS_indep(j, i)  )
#else
          while ( iv[j+1]  >TDMS_indep[j][i]  )
#endif
          {
            i++;
          }
            index[iHigh][j]=i;
            index[iLow][j]=i-1;
#ifdef UINTAH_ENABLE_KOKKOS
            distal_val[j+2]=(iv[j+1]-TDMS_indep(j, i-1))/(TDMS_indep(j, i)-TDMS_indep(j, i-1));
#else
            distal_val[j+2]=(iv[j+1]-TDMS_indep[j][i-1])/(TDMS_indep[j][i]-TDMS_indep[j][i-1]);
#endif
        }else{
#ifdef UINTAH_ENABLE_KOKKOS
            index[iHigh][j]=TDMS_d_allIndepVarNo(j+1)-1;
            index[iLow][j]=TDMS_d_allIndepVarNo(j+1)-2;
            distal_val[j+2]=(iv[j+1]-TDMS_indep(j, TDMS_d_allIndepVarNo[j+1]-2))/(TDMS_indep(j, TDMS_d_allIndepVarNo[j+1]-1)-TDMS_indep(j, TDMS_d_allIndepVarNo(j+1)-2));
#else
            index[iHigh][j]=TDMS_d_allIndepVarNo[j+1]-1;
            index[iLow][j]=TDMS_d_allIndepVarNo[j+1]-2;
            distal_val[j+2]=(iv[j+1]-TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-2])/(TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-1]-TDMS_indep[j][TDMS_d_allIndepVarNo[j+1]-2]);
#endif
        }
      }


       // ----------------perform search AGAIN for special IV  (indepednent variable-------//
       // LINEAR search
      for (int iSp=0;  iSp< oneD_switch; iSp++){
          const int cur_index=index[iSp][nDim_withSwitch-1];
#ifdef UINTAH_ENABLE_KOKKOS
        if (iv[0] <  TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo(0)-1))
#else
        if (iv[0] <  TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-1])
#endif
       {
          int i=1;
#ifdef UINTAH_ENABLE_KOKKOS
          while ( iv[0]  >TDMS_ind_1(cur_index, i)  )
#else
          while ( iv[0]  >TDMS_ind_1[cur_index][i]  )
#endif
          {
            i++;
          }
            theSpecial[iHigh][iSp]=i;
            theSpecial[iLow][iSp]=i-1;
#ifdef UINTAH_ENABLE_KOKKOS
            distal_val[iSp]=(iv[0]-TDMS_ind_1(cur_index, i-1))/(TDMS_ind_1(cur_index, i)-TDMS_ind_1(cur_index, i-1));
#else
            distal_val[iSp]=(iv[0]-TDMS_ind_1[cur_index][i-1])/(TDMS_ind_1[cur_index][i]-TDMS_ind_1[cur_index][i-1]);
#endif
        }else{
#ifdef UINTAH_ENABLE_KOKKOS
            theSpecial[iHigh][iSp]=TDMS_d_allIndepVarNo(0)-1;
            theSpecial[iLow] [iSp]=TDMS_d_allIndepVarNo(0)-2;
            distal_val[iSp]=(iv[0]-TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo(0)-2))/(TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo[0]-1)-TDMS_ind_1(cur_index, TDMS_d_allIndepVarNo(0)-2));
#else
            theSpecial[iHigh][iSp]=TDMS_d_allIndepVarNo[0]-1;
            theSpecial[iLow] [iSp]=TDMS_d_allIndepVarNo[0]-2;
            distal_val[iSp]=(iv[0]-TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-2])/(TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-1]-TDMS_ind_1[cur_index][TDMS_d_allIndepVarNo[0]-2]);
#endif
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


        for (unsigned int k = 0; k < var_index.runTime_size; k++) {
 /////      get values from table
        for (int j=0; j<npts; j++){
#ifdef UINTAH_ENABLE_KOKKOS
            table_vals[j]=TDMS_table2(var_index[k],table_indices[j]);
#else
            table_vals[j]=TDMS_table2[var_index[k]][table_indices[j]];
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

      return ;

    }



//I always thought these access functions were silly, but these actually does something; template meta progrimming
  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, 
#ifdef UINTAH_ENABLE_KOKKOS
tableContainer<Kokkos::HostSpace>
#else
tableContainer
#endif
 >::type
  getTable(){
    return table2;
  }


  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, 
#ifdef UINTAH_ENABLE_KOKKOS  
intContainer<Kokkos::HostSpace>
#else
intContainer
#endif
 >::type
  getInts(){
    return d_allIndepVarNo;
  }


  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, 
#ifdef UINTAH_ENABLE_KOKKOS
tableContainer<Kokkos::HostSpace>
#else
tableContainer
#endif
 >::type
  getSecondaryVar(){
    return indep;
  }


  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, 
#ifdef UINTAH_ENABLE_KOKKOS 
tableContainer<Kokkos::HostSpace>
#else
tableContainer
#endif
 >::type
  getPrimaryVar(){
    return ind_1;
  }


#ifdef UINTAH_ENABLE_KOKKOS
  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, tableContainer<Kokkos::HostSpace> >::type
  getPrimaryVar(){
    return ind_1;
  }


  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, tableContainer<Kokkos::HostSpace> >::type
  getSecondaryVar(){
    return indep;
  }


  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, intContainer<Kokkos::HostSpace> >::type
  getInts(){
    return d_allIndepVarNo;
  }


  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, tableContainer<Kokkos::HostSpace> >::type
  getTable(){
    return table2;
  }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, tableContainer<Kokkos::CudaSpace> >::type
  getTable(){
    return g_table2;
  }

  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, intContainer<Kokkos::CudaSpace> >::type
  getInts(){
    return g_d_allIndepVarNo;
  }

  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, tableContainer<Kokkos::CudaSpace> >::type
  getSecondaryVar(){
    return g_indep;
  }

  template<typename MemSpace>
typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, tableContainer<Kokkos::CudaSpace> >::type
  getPrimaryVar(){
    return g_ind_1;
  }
#endif

  protected:

#ifdef UINTAH_ENABLE_KOKKOS // HARD CODED TO RUN ON CPU ONLY (HOST SPACE)  and optimized for GPU (layoutLeft??)
    tableContainer<Kokkos::HostSpace> table2;          // All dependent variables
    intContainer<Kokkos::HostSpace>   d_allIndepVarNo; // size of independent variable array, for all independent variables
    tableContainer<Kokkos::HostSpace> indep;           // independent variables 1 to N-1
    tableContainer<Kokkos::HostSpace> ind_1;           // independent variable N
#else
    tableContainer table2;          // All dependent variables
    intContainer   d_allIndepVarNo; // size of independent variable array, for all independent variables
    tableContainer indep;           // independent variables 1 to N-1
    tableContainer ind_1;           // independent variable N
#endif

  public:   // avoids re-order warning

    const ClassicTableInfo tableInfo; // variable names, units, and table keys

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
  protected:
    tableContainer<Kokkos::CudaSpace> g_table2;          // All dependent variables
    intContainer<Kokkos::CudaSpace>   g_d_allIndepVarNo; // size of independent variable array, for all independent variables
    tableContainer<Kokkos::CudaSpace> g_indep;           // independent variables 1 to N-1
    tableContainer<Kokkos::CudaSpace> g_ind_1;           // independent variable N
#endif

  };
}
#endif
