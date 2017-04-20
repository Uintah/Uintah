/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//----- ClassicTableInterface.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ClassicTableInterface_h
#define Uintah_Component_Arches_ClassicTableInterface_h

#include <Core/Util/DebugStream.h>
#include <Core/IO/UintahZlibUtil.h>
#include <sstream>
#include <string>

/**
 * @class  ClassicTableInterface
 * @author Jeremy Thornock
 * @date   Jan 2011
 *
 * @brief Table interface for those created with the Classic Arches Format
 *
 * @todo
 *
 * @details
 * This class provides and interface to classic Arches formatted tables.
 * Any variable that is saved to the UDA in the dataarchiver block is automatically given a VarLabel.
 *
 * If you have trouble reading your table, you can "setenv SCI_DEBUG TABLE_DEBUG:+" to get a
 * report of what is going on in the table reader.
 *
 *
*/


namespace Uintah {
#ifdef UINTAH_ENABLE_KOKKOS
typedef Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::RandomAccess> > tempTableContainer;
typedef Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::RandomAccess> > tableContainer ;
#else
typedef std::vector<std::vector<double> > tempTableContainer;
typedef const std::vector<std::vector<double> >& tableContainer ;
#endif


class ArchesLabel;
class MPMArchesLabel;
class TimeIntegratorLabel;
class BoundaryCondition_new;
class MixingRxnModel;

class ClassicTableInterface : public MixingRxnModel {

public:

  ClassicTableInterface( SimulationStateP& sharedState );

  ~ClassicTableInterface();

  void problemSetup( const ProblemSpecP& params );

  /** @brief Gets the thermochemical state for a patch
      @param initialize         Tells the method to allocateAndPut
      @param modify_ref_den     Tells the method to modify the reference density */
  void sched_getState( const LevelP& level,
                       SchedulerP& sched,
                       const int time_substep,
                       const bool initialize,
                       const bool modify_ref_den );

  /** @brief Gets the thermochemical state for a patch
      @param initialize         Tells the method to allocateAndPut
      @param modify_ref_den     Tells the method to modify the reference density */
  void getState( const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw,
                 const int time_substep,
                 const bool initialize,
                 const bool modify_ref_den );

  /** @brief Load table into memory */
  void loadMixingTable(gzFile &fp, const std::string & inputfile );
  void loadMixingTable(std::stringstream& table_stream,
                       const std::string & inputfile );

  enum BoundaryType { DIRICHLET, NEUMANN, FROMFILE };

  struct DepVarCont {

    CCVariable<double>* var;
    int index;

  };

  /** @brief returns the heat loss bounds from the table **/
  inline std::vector<double> get_hl_bounds(){
    std::vector<double> bounds;
    bounds.push_back(d_hl_lower_bound);
    bounds.push_back(d_hl_upper_bound);
    return bounds; };


  /*********interp derived classes*****************************************/

  /** @brief A base class for Interpolation */
  class Interp_class {

  public:

    Interp_class( tableContainer  table,
                  const std::vector<int>& IndepVarNo,
                  const std::vector<std::vector<double> > & indepin,
                  const std::vector<std::vector<double> >& ind_1in )
      : table2(table), d_allIndepVarNo(IndepVarNo), indep(indepin), ind_1(ind_1in)
    {}

    virtual ~Interp_class() {}

    virtual inline std::vector<double> find_val( const std::vector<double>& iv, const std::vector<int>& var_index) {return std::vector<double>(1,0.0);}

  protected:

    tableContainer  table2;
    const std::vector<int>&  d_allIndepVarNo;
    const std::vector< std::vector <double> >&  indep;
    const std::vector< std::vector <double > >&  ind_1;

    //std::mutex d_interpLock; // For synchronization in find_val() functions

  };

  class Interp1 : public Interp_class {

  public:

    Interp1( const std::vector<int>& indepVarNo, tableContainer  table,
             const std::vector< std::vector <double> >& i1)
      : Interp_class(table, indepVarNo, i1, i1 ) {
    }

    ~Interp1() {};

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
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1)
      : Interp_class( table, indepVarNo, indep_headers, i1 ){}

    ~Interp2() {}

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
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1)
      : Interp_class( table, indepVarNo, indep_headers, i1 ) {}

    ~Interp3() {}

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
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1)
      : Interp_class(table, indepVarNo, indep_headers, i1 ){}

    ~Interp4(){}

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
            npts = (int)pow(2.0,i);
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
             const std::vector< std::vector <double> >& indep_headers, const std::vector< std::vector <double > >& i1, int d_indepvarscount)
      : Interp_class( table, indepVarNo, indep_headers, i1 ){

      multiples = std::vector<int>(d_indepvarscount);
      multtemp = 0;
      for (int i = 0; i < d_indepvarscount; i++) {
        multtemp = 1;
        for (int j = 0; j<i; j++) {
          multtemp = multtemp * indepVarNo[j];
        }
        multiples[i] = multtemp;
      }

      int npts = (int)pow(2.0,d_indepvarscount);
      value_pop = std::vector< std::vector <bool> > (npts);

      for (int i =0; i < npts; i++) {
        value_pop[i] = std::vector<bool>(d_indepvarscount );
      }

      //bool matrix for use in lookup
      int temp_pts;
      double temp_pts_d;
      for (int i=0; i < npts; i++) {
        for (int j = d_indepvarscount-1; j >= 0; j--) {
          temp_pts_d = pow(2.0, j);
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

    ~InterpN(){}

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

        npts = (int)pow(2.0,ivcount);
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
            npts = (int)pow(2.0,i);
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

  //*******************************************end interp classes//

  typedef std::map<std::string, DepVarCont >       DepVarMap;
  typedef std::map<std::string, int >               IndexMap;

  /** @brief Return a table lookup for a variable given the independent variable space. **/
  double getTableValue( std::vector<double>, std::string );

  /** @brief Match the requested dependent variable with their table index. **/
  void tableMatching();

  /** @brief Return a table lookup for a variable given the independent variables and set of inerts (may be an empty set) - Grid wise **/
  double getTableValue( std::vector<double> iv, std::string depend_varname, StringToCCVar inert_mixture_fractions, IntVector c);

  /** @brief Return a table lookup for a variable given the independent variables and set of inerts (may be an empty set) - single point wise**/
  double getTableValue( std::vector<double> iv, std::string depend_varname, doubleMap inert_mixture_fractions );

  /** @brief Return a reference to the inert map for scheduling purposes in other classes.**/
  InertMasterMap& getInertMap(){
    return d_inertMap;
  };

  /** @brief Method to find the index for any dependent variable.  **/
  int inline findIndex( std::string name ){

    int index = -1;

    for ( int i = 0; i < d_varscount; i++ ) {

      if ( name.compare( d_allDepVarNames[i] ) == 0 ) {
        index = i;
        break;
      }
    }

    if ( index == -1 ) {
      std::ostringstream exception;
      exception << "Error: The variable " << name << " was not found in the table." << "\n" <<
        "Please check your input file and try again. " << std::endl;
      throw InternalError(exception.str(),__FILE__,__LINE__);
    }

    return index;
  }

private:

  Interp_class * ND_interp;

  bool d_table_isloaded;    ///< Boolean: has the table been loaded?

  // Specifically for the classic table:
  double d_f_stoich;        ///< Stoichiometric mixture fraction
  double d_H_fuel;          ///< Fuel Enthalpy
  double d_H_air;           ///< Oxidizer Enthalpy
  double d_hl_lower_bound;  ///< Heat loss lower bound
  double d_hl_upper_bound;  ///< Heat loss upper bound
  double d_wall_temp;       ///< Temperature at a domain wall


  int d_indepvarscount;     ///< Number of independent variables
  int d_varscount;          ///< Total dependent variables
  int d_nDepVars;           ///< number of dependent variables requested by arches

  std::string d_enthalpy_name;
  const VarLabel* d_enthalpy_label;

  IndexMap d_depVarIndexMap;                      ///< Reference to the integer location of the variable
  IndexMap d_enthalpyVarIndexMap;                 ///< Reference to the integer location of variables for heat loss calculation

  std::vector<int>    d_allIndepVarNum;         ///< Vector storing the grid size for the Independent variables
  std::vector<std::string> d_allDepVarUnits;         ///< Units for the dependent variables
  std::vector<std::string> d_allUserDepVarNames;     ///< Vector storing all independent variable names requested in input file

  BoundaryCondition_new* _boundary_condition;

  void checkForConstants(gzFile &fp, const std::string & inputfile );
  void checkForConstants(std::stringstream& table_stream,
                         const std::string & inputfile );

  //previous Arches specific variables:
  std::vector<std::vector<double> > i1;

#ifdef UINTAH_ENABLE_KOKKOS
  // magic of kokkos, don't pass by reference and when an object goes out of scope it isn't necessarily deleted
#else
  tempTableContainer table;
#endif

  std::vector<std::vector<double> > indep_headers;

  /// A dependent variable wrapper
  struct ADepVar {
    std::string name;
    CCVariable<double> data;
  };


  void getIndexInfo();
  void getEnthalpyIndexInfo();

}; // end class ClassicTableInterface
} // end namespace Uintah

#endif
