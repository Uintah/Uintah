/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- ClassicTableInterface.h --------------------------------------------------

#ifndef Uintah_Component_Arches_ClassicTableInterface_h
#define Uintah_Component_Arches_ClassicTableInterface_h

#include <CCA/Components/Arches/Mixing/InletStream.h>
#include <CCA/Components/Arches/Mixing/Stream.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Core/Util/DebugStream.h>

#include   <string>

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
 
This code checks for the following tags/attributes in the input file:
The UPS interface is: 

\code
<ClassicTable                   spec="OPTIONAL NO_DATA">
  <inputfile                    spec="REQUIRED STRING" /> <!-- table to be opened --> 
  <cold_flow                    spec="OPTIONAL BOOLEAN"/> <!-- use for simple two stream mixing --> 
  <noisy_hl_warning             spec="OPTIONAL NO_DATA"/> <!-- warn when heat loss is clipped to bounds --> 
  <hl_scalar_init               spec="OPTIONAL DOUBLE" /> <!-- initial heat loss value in the domain --> 
  <coal                         spec="OPTIONAL NO_DATA" 
                                attribute1="fp_label REQUIRED STRING"     
                                attribute2="eta_label REQUIRED STRING"/> 
                                <!-- Attributes must match the transported IVs specified in the TransportEqn node --> 
</ClassicTable>

<DataArchiver>
    <save name=STRING table_lookup="true"> <!-- note that STRING must match the name in the table -->
</DataArchiver>
\endcode

 * Any variable that is saved to the UDA in the dataarchiver block is automatically given a VarLabel.  
 *
 * If you have trouble reading your table, you can "setenv SCI_DEBUG TABLE_DEBUG:+" to get a 
 * report of what is going on in the table reader.
 *
 *
*/


namespace Uintah {

class ArchesLabel; 
class MPMArchesLabel; 
class TimeIntegratorLabel; 
class BoundaryCondition_new; 
class ClassicTableInterface : public MixingRxnModel {

public:

  ClassicTableInterface( const ArchesLabel* labels, const MPMArchesLabel* MAlabels );

  ~ClassicTableInterface();

  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Gets the thermochemical state for a patch 
      @param initialize         Tells the method to allocateAndPut 
      @param with_energy_exch   Tells the method that energy exchange is on
      @param modify_ref_den     Tells the method to modify the reference density */
  void sched_getState( const LevelP& level, 
                       SchedulerP& sched, 
                       const TimeIntegratorLabel* time_labels, 
                       const bool initialize,
                       const bool with_energy_exch,
                       const bool modify_ref_den ); 

  /** @brief Gets the thermochemical state for a patch 
      @param initialize         Tells the method to allocateAndPut 
      @param with_energy_exch   Tells the method that energy exchange is on
      @param modify_ref_den     Tells the method to modify the reference density */
  void getState( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw, 
                 const TimeIntegratorLabel* time_labels, 
                 const bool initialize, 
                 const bool with_energy_exch, 
                 const bool modify_ref_den );

  /** @brief Schedule computeHeatLoss */
  void sched_computeHeatLoss( const LevelP& level, 
                              SchedulerP& sched, 
                              const bool intialize_me, const bool calcEnthalpy ); 

  /** @brief  Computes the heat loss from the table */
  void computeHeatLoss( const ProcessorGroup* pc, 
                        const PatchSubset* patches, 
                        const MaterialSubset* matls, 
                        DataWarehouse* old_dw, 
                        DataWarehouse* new_dw, 
                        const bool intialize_me, 
                        const bool calcEnthalpy ); 


  /** @brief A temporary solution to deal with boundary conditions on properties until Properties.cc is eliminated */ 
  void oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type );

  /** @brief  schedules computeFirstEnthalpy */
  void sched_computeFirstEnthalpy( const LevelP& level, SchedulerP& sched ); 

  /** @brief This will initialize the enthalpy to a table value for the first timestep */
  void computeFirstEnthalpy( const ProcessorGroup* pc, 
                             const PatchSubset* patches, 
                             const MaterialSubset* matls, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw ); 

  /** @brief    Load list of dependent variables from the table 
      @returns  A vector<string>& that is a reference to the list of all dependent variables */
  const vector<string> & getAllDepVars();

  /** @brief    Load list of independent variables from the table
      @returns  A vector<string>& that is a reference to the list of all independent variables */ 
  const vector<string> & getAllIndepVars();

  /** @brief Dummy initialization as required by MPMArches */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief Dummy initialization as required by MPMArches */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /** @brief Load table into memory */ 
  void loadMixingTable( const string & inputfile );

  enum BoundaryType { DIRICHLET, NEUMANN };

  struct DepVarCont {

    CCVariable<double>* var; 
    int index; 

  }; 
	

	/*********interp derived classes*****************************************/
	
  /** @brief A base class for Interpolation */
  class Interp_class {

	public:
		Interp_class(){};
		virtual ~Interp_class(){};
	
		virtual inline double find_val( std::vector<double> iv, int var_index)

		{return 0;};
	
	protected:

		std::vector<int>  d_allIndepVarNo;
		std::vector<double> table_vals;
		std::vector<std::vector<double> >  table2;
		std::vector< std::vector <double> >  indep;
		std::vector< std::vector <double > >  ind_1;
		std::vector<int> lo_index;
		std::vector<int> hi_index;
  };

	class Interp1 : public Interp_class {
		
	public:
		Interp1(std::vector<int> d_allIndepVarNum, std::vector<std::vector<double> > table, 
						std::vector< std::vector <double> > i1) {
			d_allIndepVarNo = d_allIndepVarNum;
			ind_1 = i1;
			table2 = table;
			
			table_vals = vector<double>(2);
			lo_index = vector<int>(1);
			hi_index = vector<int>(1);
			};

		~Interp1(){};
		
		inline double find_val( std::vector <double> iv, int var_index) {
			
			int i1dep_ind = 0;     
			int mid = 0;
			int lo_ind = 0;
			int hi_ind = d_allIndepVarNo[0] - 1;
			double iv_val = iv[0];
			
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
			
			table_vals[0] = table2[var_index][lo_index[0]];
			table_vals[1] = table2[var_index][hi_index[0]];
			
			double var_val = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[0][lo_index[0]])*(iv[0]-ind_1[0][lo_index[0]])+ table_vals[0];
			return var_val;
			
		};
	};
	
	class Interp2 : public Interp_class {
	
	public:

		Interp2(std::vector<int> d_allIndepVarNum,std::vector<std::vector<double> > table,
						std::vector< std::vector <double> > indep_headers,std::vector< std::vector <double > > i1){
			d_allIndepVarNo = d_allIndepVarNum;
			indep = indep_headers;
			ind_1 = i1;
			table2 = table;
			
			table_vals = vector<double>(4);
			lo_index = vector<int>(2);
			hi_index = vector<int>(2);
		
		};
		~Interp2(){};
		
		inline double find_val( std::vector<double> iv, int var_index)

		{
			
			int mid = 0;
      int lo_ind;
			int hi_ind;
			double iv_val;
						
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
					lo_ind = hi_index[i]-1;  
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
			
			table_vals[0] = table2[var_index][d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
			table_vals[1] = table2[var_index][d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
			table_vals[2] = table2[var_index][d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
			table_vals[3] = table2[var_index][d_allIndepVarNo[0] * hi_index[1] + hi_index[0]]; 
			
			table_vals[0] = (table_vals[2] - table_vals[0])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[0];
			table_vals[1] = (table_vals[3] - table_vals[1])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[1];
			
			double var_val = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+ table_vals[0];
			return var_val;
			
		};
	protected:	
		
	}; 
	
	class Interp3 : public Interp_class {
		
	public:
		Interp3(std::vector<int> d_allIndepVarNum,std::vector<std::vector<double> > table,
						std::vector< std::vector <double> > indep_headers,std::vector< std::vector <double > > i1){
		  d_allIndepVarNo = d_allIndepVarNum;
			indep = indep_headers;
			ind_1 = i1;
			table2 = table;
	 
			table_vals = vector<double>(8);
			lo_index = vector<int>(3);
			hi_index = vector<int>(3);
		};
		~Interp3(){};
		
		inline double find_val( std::vector<double> iv, int var_index)

		{
			int mid = 0;
			int lo_ind;
			int hi_ind;
			double iv_val;
			bool Ncheck = false;
			
			//binary search loop 2-> N
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
							//if (indep_headers[i-1][mid] ==  iv[i])
							lo_ind = mid;
							hi_ind = mid;
						} 
					}			   
				} else if (indep[i-1][lo_ind] == iv_val) {
					hi_ind = 1; 
				} else {
					lo_ind = hi_index[i]-1;  
					if (i == 2) {
						Ncheck = true;
					}
					
				}
				lo_index[i] = lo_ind;
				hi_index[i] = hi_ind;
				
				if (iv_val < indep[i-1][0]) {
					lo_index[i] = 0;
					hi_index[i] = 0;
				}
			}
			int i1dep_ind = 0;
			//binary search for i1
			if (Ncheck) {
				i1dep_ind = hi_index[2];
			} else {
			 i1dep_ind = lo_index[2];     //assume i1 is dep on last var
			}
				
				
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

			table_vals[0] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
			table_vals[1] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
			table_vals[2] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
			table_vals[3] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]]; 
			table_vals[4] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
			table_vals[5] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
			table_vals[6] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
			table_vals[7] = table2[var_index][d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
			
						table_vals[0] = (table_vals[4] - table_vals[0])/(indep[1][lo_index[2]+1]-indep[1][lo_index[2]])*(iv[2]-indep[1][lo_index[2]]) + table_vals[0]; 
			table_vals[1] = (table_vals[5] - table_vals[1])/(indep[1][lo_index[2]+1]-indep[1][lo_index[2]])*(iv[2]-indep[1][lo_index[2]]) + table_vals[1];
			table_vals[2] = (table_vals[6] - table_vals[2])/(indep[1][lo_index[2]+1]-indep[1][lo_index[2]])*(iv[2]-indep[1][lo_index[2]]) + table_vals[2];
			table_vals[3] = (table_vals[7] - table_vals[3])/(indep[1][lo_index[2]+1]-indep[1][lo_index[2]])*(iv[2]-indep[1][lo_index[2]]) + table_vals[3];
			
			table_vals[0] = (table_vals[2]-table_vals[0])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[0]; 
			table_vals[1] = (table_vals[3]-table_vals[1])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[1];

			double var_val = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[0];
			/*
			table_vals[0] = (table_vals[1] - table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[0];
			table_vals[1] = (table_vals[3] - table_vals[2])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[2];
			table_vals[2] = (table_vals[5] - table_vals[4])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[4];                                                                                                                
			table_vals[3] = (table_vals[7] - table_vals[6])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[6];

			table_vals[0] = (table_vals[1]-table_vals[0])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[0];
			table_vals[1] = (table_vals[3]-table_vals[2])/(indep[0][lo_index[1]+1]-indep[0][lo_index[1]])*(iv[1]-indep[0][lo_index[1]]) + table_vals[2];
      
			double var_val = (table_vals[1] - table_vals[0])/(indep[1][lo_index[2]+1]-indep[1][lo_index[2]])*(iv[2]-indep[1][lo_index[2]]) + table_vals[0];
			*/

			return var_val;
		};
	}; 
	
	class Interp4 : public Interp_class {
	public:
		Interp4(std::vector<int> d_allIndepVarNum,std::vector<std::vector<double> > table,
						std::vector< std::vector <double> > indep_headers,std::vector< std::vector <double > > i1){

			d_allIndepVarNo = d_allIndepVarNum;
			indep = indep_headers;
			ind_1 = i1;
			table2 = table;
			
			table_vals = vector<double>(16);
			lo_index = vector<int>(4);
			hi_index = vector<int>(4);
		};
		~Interp4(){};
		
		inline double find_val(std::vector<double> iv, int var_index)
		{
			int mid = 0;
			double var_value = 0.0;
			int lo_ind;
			int hi_ind;
			double iv_val;
			
			//binary search loop 2-> N
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
					lo_ind = hi_index[i]-1;  
				}
				lo_index[i] = lo_ind;
				hi_index[i] = hi_ind;
				
				if (iv_val < indep[i-1][0]) {
					lo_index[i] = 0;
					hi_index[i] = 0;
				}
			}
			
			//binary search for i1
			int i1dep_ind = lo_index[3];     //assume i1 is dep on last var
			
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
			
			//popvals
			table_vals[0] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
			table_vals[1] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
			table_vals[2] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
			table_vals[3] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]]; 
			table_vals[4] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
			table_vals[5] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
			table_vals[6] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
			table_vals[7] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
			table_vals[8] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
			table_vals[9] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
			table_vals[10] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
			table_vals[11] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*lo_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]]; 
			table_vals[12] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + lo_index[0]];
			table_vals[13] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * lo_index[1] + hi_index[0]];
			table_vals[14] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + lo_index[0]];
			table_vals[15] = table2[var_index][d_allIndepVarNo[2]*d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[3]+d_allIndepVarNo[1]*d_allIndepVarNo[0]*hi_index[2] + d_allIndepVarNo[0] * hi_index[1] + hi_index[0]];
			
			int npts =0;
			for (int i = 3; i > 0; i--) {	
				npts = pow(2.0,i);
				for (int k=0; k < npts; k++) {
					table_vals[k] = (table_vals[k+npts]-table_vals[k])/(indep[i-1][lo_index[i]+1]-indep[i-1][lo_index[i]])*(iv[i]-indep[i-1][lo_index[i]])+table_vals[k];
				}
			}
			
			table_vals[0] = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[0];
			var_value = table_vals[0];
			return var_value;
			
		};
	};
	
	class InterpN : public Interp_class {
	  public:
		InterpN(std::vector<int> d_allIndepVarNum,std::vector<std::vector<double> > table,
						std::vector< std::vector <double> > indep_headers,std::vector< std::vector <double > > i1, int d_indepvarscount)
		{
		  multiples = vector<int>(d_indepvarscount);
		  multtemp = 0;
		  for (int i = 0; i < d_indepvarscount; i++) {
			  multtemp = 1;
			  for (int j = 0; j<i; j++) {
				  multtemp = multtemp * d_allIndepVarNum[j];
			  }
			  multiples[i] = multtemp;
		  }
  	
		  int npts = pow(2.0,d_indepvarscount);
		  value_pop = vector< vector <bool> > (npts);
		
		  for (int i =0; i < npts; i++) {
			  value_pop[i] = vector<bool>(d_indepvarscount );
		  }
		
		  //bool matrix for use in lookup
		  int temp_pts;
		  double temp_pts_d;
		  for (int i=0; i < npts; i++) {
			  for (int j = d_indepvarscount-1; j >= 0; j--) {
				  temp_pts_d = pow(2.0, j);
				  temp_pts = floor((i/temp_pts_d));
				  if ((temp_pts % 2) == 0) {
					  value_pop[i][j] = true;
				  } else {
					  value_pop[i][j] = false;
				  }
			  }
		  }
			
			d_allIndepVarNo = d_allIndepVarNum;
			indep = indep_headers;
			ind_1 = i1;
			table2 = table;
			
			table_vals = vector<double>(npts);
			lo_index = vector<int>(d_indepvarscount);
			hi_index = vector<int>(d_indepvarscount);
			ivcount = d_indepvarscount;
			
		};
		~InterpN(){};
		
		inline double find_val(std::vector<double> iv, int var_index)
		{
			
			int mid = 0;
			double var_value = 0.0;
			int lo_ind;
			int hi_ind;
			double iv_val;
			 
			//binary search loop 2-> N
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
			int i1dep_ind = lo_index[ivcount-1];     //assume i1 is dep on last var
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
			
			int npts = 0; 

			npts = pow(2.0,ivcount);
			int tab_index;
			
			//interpolant loop - 2parts read-in & calc
			for (int i=0; i < npts; i++) {
				tab_index = 0;
				for (int j = ivcount-1; j >= 0; j--) {
					if (value_pop[i][j]) { //determines hi/lo on bool
						tab_index = tab_index + multiples[j]*lo_index[j];
					} else {
						tab_index = tab_index + multiples[j]*hi_index[j];
					}
				}
				table_vals[i] = table2[var_index][tab_index];
			} 
			 
			for (int i = ivcount-1; i > 0; i--) {
				npts = pow(2.0,i);
				for (int k=0; k < npts; k++) {
					table_vals[k] = (table_vals[k+npts]-table_vals[k])/(indep[i-1][lo_index[i]+1]-indep[i-1][lo_index[i]])*(iv[i]-indep[i-1][lo_index[i]])+table_vals[k];
				}
			}
			
			table_vals[0] = (table_vals[1]-table_vals[0])/(ind_1[i1dep_ind][lo_index[0]+1]-ind_1[i1dep_ind][lo_index[0]])*(iv[0]-ind_1[i1dep_ind][lo_index[0]])+table_vals[0];
			
			var_value = table_vals[0];
			return var_value;
			
		}
		
	  protected:
		int ivcount;
		std::vector<int> multiples;
		std::vector <std::vector <bool> > value_pop;
		int multtemp;
	};
	
  //*******************************************end interp classes//
  
  typedef std::map<string, DepVarCont >       DepVarMap;
  typedef std::map<string, int >               IndexMap; 

  double getTableValue( std::vector<double>, std::string ); 

	void tableMatching(); 
	
protected :
	
private:

	Interp_class * ND_interp;
	
  bool d_table_isloaded;    ///< Boolean: has the table been loaded?
  bool d_noisy_hl_warning;  ///< Provide information about heat loss clipping
  bool d_allocate_soot;     ///< For new DORadiation source term...allocate soot variable 
  bool _use_mf_for_hl;     ///< Rather than using adiabatic enthalpy from the table, compute using mix. frac and fuel/ox enthalpy

  double d_hl_scalar_init;  ///< Heat loss value for non-adiabatic conditions
  // Specifically for the classic table: 
  double d_f_stoich;        ///< Stoichiometric mixture fraction 
  double d_H_fuel;          ///< Fuel Enthalpy
  double d_H_air;           ///< Oxidizer Enthalpy
  double d_hl_lower_bound;  ///< Heat loss lower bound
  double d_hl_upper_bound;  ///< Heat loss upper bound
  double d_wall_temp;       ///< Temperature at a domain wall 

  
  int d_indepvarscount;     ///< Number of independent variables
  int d_varscount;          ///< Total dependent variables

  string d_enthalpy_name; 
  const VarLabel* d_enthalpy_label; 

  IntVector d_ijk_den_ref;                ///< Reference density location

  IndexMap d_depVarIndexMap;              ///< Reference to the integer location of the variable
  IndexMap d_enthalpyVarIndexMap;         ///< Referece to the integer location of variables for heat loss calculation

  std::vector<int>    d_allIndepVarNum;        ///< Vector storing the grid size for the Independant variables
  std::vector<string> d_allDepVarNames;        ///< Vector storing all dependent variable names from the table file
  std::vector<string> d_allDepVarUnits;        ///< Units for the dependent variables 

  vector<string> d_allUserDepVarNames;    ///< Vector storing all independent varaible names requested in input file

  BoundaryCondition_new* _boundary_condition; 

  void checkForConstants( const string & inputfile );

  //previous Arches specific variables: 
  std::vector<std::vector<double> > i1; 
  std::vector <std::vector <double> > table;
	
  std::vector <std::vector <double> > indep_headers;

  /// A dependent variable wrapper
  struct ADepVar {
    string name; 
    CCVariable<double> data; 
  };


  /// @brief Method to find the index for any dependent variable.  
  int inline findIndex( std::string name ){ 

    int index = -1; 

    for ( int i = 0; i < d_varscount; i++ ) { 

      if ( name.compare( d_allDepVarNames[i] ) == 0 ) {
        index = i; 
        break; 
      }
    }

    if ( index == -1 ) {
      ostringstream exception;
      exception << "Error: The variable " << name << " was not found in the table." << "\n" << 
        "Please check your input file and try again. " << endl;
      throw InternalError(exception.str(),__FILE__,__LINE__);
    }

    return index; 
  }

  void getIndexInfo(); 
  void getEnthalpyIndexInfo(); 

}; // end class ClassicTableInterface
} // end namespace Uintah

#endif
