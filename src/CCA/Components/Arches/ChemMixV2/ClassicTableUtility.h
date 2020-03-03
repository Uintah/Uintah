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

#ifndef Uintah_Component_Arches_ClassicTableUtilities_h
#define Uintah_Component_Arches_ClassicTableUtilities_h

/**
 * @class  ClassicTable
 * @author Jeremy Thornock, Derek Harris (2017)
 * @date   Jan 2011
 *
 * @brief functions to populate table and return pointer to the created table.  
 *
 * @todo
 *
 * @details
 * This header was created to avoid compiler warnings, due to the included functions not belonging to a class.
 * RECOMMENDED USE: Only include this header file in .cc files.  May not work with old_table compiler flag.
*/


#include <CCA/Components/Arches/ChemMixV2/ClassicTable.h>
#include <Core/IO/UintahZlibUtil.h>
#include <sci_defs/kokkos_defs.h>


namespace Uintah {

template<unsigned int max_dep_request_at_a_time, class fileTYPE>
Interp_class<max_dep_request_at_a_time>*
loadMixingTable(fileTYPE &fp, const std::string & inputfile,  std::vector<std::string> &d_savedDep_var ,  std::map<std::string,double> &d_constants)  
{

  int  loadAll=false; // load all dependent variables?
  if (d_savedDep_var.size()==0)
    loadAll=true;

  std::vector<std::string> d_allIndepVarNames;     ///< Vector storing all independent variable names from table file
  std::vector<std::string> d_allDepVarUnits;       ///< Units for the dependent variables

  proc0cout << " Preparing to load the table from inputfile:   " << inputfile << "\n";
  int d_indepvarscount = getInt( fp );

  proc0cout << " Total number of independent variables: " << d_indepvarscount << std::endl;

  d_allIndepVarNames = std::vector<std::string>(d_indepvarscount);

#ifdef UINTAH_ENABLE_KOKKOS
  tempIntContainer<Kokkos::HostSpace> d_allIndepVarNum("array_of_ind_var_sizes",d_indepvarscount);  //< std::vector storing the grid size for the Independent variables
#else
  std::vector<int> *d_allIndepVarNum=scinew std::vector<int>(d_indepvarscount);                     //< std::vector storing the grid size for the Independent variables
#endif

  for (int ii = 0; ii < d_indepvarscount; ii++){
    std::string varname = getString( fp );
    d_allIndepVarNames[ii] = varname;
  }
  for (int ii = 0; ii < d_indepvarscount; ii++){
    int grid_size = getInt( fp );
#ifdef UINTAH_ENABLE_KOKKOS
    d_allIndepVarNum(ii) = grid_size;
#else
    (*d_allIndepVarNum)[ii] = grid_size;
#endif
  }

  int d_varscount = getInt( fp );
  proc0cout << " Total dependent variables in table: " << d_varscount << std::endl;

   std::vector<std::string> allDepVarNames(d_varscount);
  for (int ii = 0; ii < d_varscount; ii++) {

    std::string variable;
    variable = getString( fp );
    allDepVarNames[ii] = variable ;

  }



  std::vector<int> index_map(d_varscount,-1);
  if (loadAll){
      for (int ii = 0; ii < d_varscount; ii++) {
          index_map[ii]=ii;
         d_savedDep_var.push_back(allDepVarNames[ii]);
      }
  }else{
  for (unsigned int ix=0; ix <  d_savedDep_var.size(); ix++){
      for (int ii = 0; ii < d_varscount; ii++) {
        if ( allDepVarNames[ii] ==d_savedDep_var[ix]){
          index_map[ii]=ix;
          break;
        }
        if (ii==d_varscount-1){
          throw ProblemSetupException( std::string("requested dependent variable "+ d_savedDep_var[ix] + " not found in table. ") , __FILE__, __LINE__ );
        }
      }
    }
  }

  // Units
  d_allDepVarUnits = std::vector<std::string>(d_varscount);
  for (int ii = 0; ii < d_varscount; ii++) {
    std::string units = getString( fp );
    d_allDepVarUnits[ii] =  units ;
  }

  //indep vars grids


#ifdef UINTAH_ENABLE_KOKKOS
    int max_size=0;
    for (int i = 0; i < d_indepvarscount - 1; i++) {
      max_size=max(max_size, d_allIndepVarNum(i+1)); // pad this non-square portion of the table = (
    }
    tempTableContainer<Kokkos::HostSpace> indep_headers("secondary_independent_variables",d_indepvarscount-1,max_size);

    tempTableContainer<Kokkos::HostSpace> i1("primary_independent_variable",d_allIndepVarNum(d_indepvarscount-1),d_allIndepVarNum(0));
#else
    std::vector<std::vector<double> > *indep_headers = scinew std::vector<std::vector<double> >(d_indepvarscount-1);  //vector contains 2 -> N dimensions
    for (int i = 0; i < d_indepvarscount - 1; i++) {
      (*indep_headers)[i] = std::vector<double>((*d_allIndepVarNum)[i+1]);
    }

    std::vector<std::vector<double> > *i1=scinew std::vector<std::vector<double> >((*d_allIndepVarNum)[d_indepvarscount-1], std::vector<double>((*d_allIndepVarNum)[0])  );
#endif

  //assign values (backwards)
  for (int i = d_indepvarscount-2; i>=0; i--) {
#ifdef UINTAH_ENABLE_KOKKOS
    for (int j = 0; j < d_allIndepVarNum(i+1) ; j++) 
#else
    for (int j = 0; j < (*d_allIndepVarNum)[i+1] ; j++) 
#endif
    {
      double v = getDouble( fp );
#ifdef UINTAH_ENABLE_KOKKOS
      indep_headers(i, j) = v;
#else
      (*indep_headers)[i][j] = v;
#endif
    }
  }

  int size=1;
  //ND size
  for (int i = 0; i < d_indepvarscount; i++) {
#ifdef UINTAH_ENABLE_KOKKOS
    size = size*d_allIndepVarNum(i);
#else
    size = size*(*d_allIndepVarNum)[i];
#endif
  }
  int num_dep_vars=loadAll ? d_varscount : d_savedDep_var.size();
#ifdef UINTAH_ENABLE_KOKKOS
    tempTableContainer<Kokkos::HostSpace> table("ClassicMixingTable",num_dep_vars,size);
#else
    tempTableContainer* table=new tempTableContainer(num_dep_vars , std::vector<double> (size,0.0));
#endif

#ifdef UINTAH_ENABLE_KOKKOS
  int size2 = size/d_allIndepVarNum(d_indepvarscount-1);
#else
  int size2 = size/(*d_allIndepVarNum)[d_indepvarscount-1];
#endif
  proc0cout << "Table size " << size << std::endl;

  proc0cout << "Reading in the dependent variables: " << std::endl;
  bool read_assign = true;
    for (int kk = 0; kk < d_varscount; kk++) {
      
        if ( index_map[kk] <0){
          proc0cout << " skipping---> " << allDepVarNames[kk] << std::endl;
        }else{
          proc0cout << " loading ---> " << allDepVarNames[kk] << std::endl;
        }


#ifdef UINTAH_ENABLE_KOKKOS
        for (int mm = 0; mm < d_allIndepVarNum(d_indepvarscount-1); mm++) 
#else
        for (int mm = 0; mm < (*d_allIndepVarNum)[d_indepvarscount-1]; mm++) 
#endif
        {
          if (read_assign) {
#ifdef UINTAH_ENABLE_KOKKOS
            for (int i = 0; i < d_allIndepVarNum(0); i++)
#else
            for (int i = 0; i < (*d_allIndepVarNum)[0]; i++)
#endif
            {
              double v = getDouble(fp);
              if ( index_map[kk] >-1){
#ifdef UINTAH_ENABLE_KOKKOS
              i1(mm,i) = v;
#else
              (*i1)[mm][i] = v;
#endif
              }
            }
          } else {
            //read but don't assign inbetween vals
#ifdef UINTAH_ENABLE_KOKKOS
            for (int i = 0; i < d_allIndepVarNum(0); i++) {
#else
            for (int i = 0; i < (*d_allIndepVarNum)[0]; i++) {
#endif
              getDouble(fp);
            }
          }
          for (int j=0; j<((d_indepvarscount > 1) ? size2 : size); j++) { // since 1D is a special case
            double v = getDouble(fp);
            if ( index_map[kk] >-1){
#ifdef UINTAH_ENABLE_KOKKOS
              table(index_map[kk],j + mm*size2) = v;
#else
              (*table)[index_map[kk] ][j + mm*size2] = v;
#endif
            }
          }
          if (d_indepvarscount == 1 ) {  // since 1-D is a special case
            break;
          }
        }
      if ( read_assign ) { read_assign = false; }
    }

#ifdef UINTAH_ENABLE_KOKKOS
        ClassicTableInfo infoStruct(indep_headers, d_allIndepVarNum, d_allIndepVarNames,d_savedDep_var, d_allDepVarUnits,d_constants); 
    return   scinew Interp_class<max_dep_request_at_a_time>( table,d_allIndepVarNum, indep_headers, i1,infoStruct);
#else
    ClassicTableInfo infoStruct(*indep_headers, *d_allIndepVarNum, d_allIndepVarNames,d_savedDep_var, d_allDepVarUnits,d_constants); 

    return   scinew Interp_class<max_dep_request_at_a_time>(*table,*d_allIndepVarNum, *indep_headers, *i1,infoStruct);
#endif

}
#ifdef OLD_TABLE
static
void
checkForConstants(gzFile &fp, const std::string & inputfile, std::map<std::string,double> &d_constants ) {

  proc0cout << "\n Looking for constants in the header... " << std::endl;

  bool look = true;
  while ( look ){

    char ch = gzgetc( fp );

    if ( ch == '#' ) {

      char key = gzgetc( fp );

      if ( key == 'K' ) {
        for (int i = 0; i < 3; i++ ){
          key = gzgetc( fp ); // reading the word KEY and space
        }

        std::string name;
        while ( true ) {
          key = gzgetc( fp );
          if ( key == '=' ) {
            break;
          }
          name.push_back( key );  // reading in the token's key name
        }

        std::string value_s;
        while ( true ) {
          key = gzgetc( fp );
          if ( key == '\n' || key == '\t' || key == ' ' ) {
            break;
          }
          value_s.push_back( key ); // reading in the token's value
        }

        double value;
        sscanf( value_s.c_str(), "%lf", &value );

        proc0cout << " KEY found: " << name << " = " << value << std::endl;

        d_constants.insert( make_pair( name, value ) );

      } else {
        while ( true ) {
          ch = gzgetc( fp ); // skipping this line
          if ( ch == '\n' || ch == '\t' ) {
            break;
          }
        }
      }

    } else {

      look = false;

    }
  }
}
#else
static
void
checkForConstants(std::stringstream &table_stream,
    const std::string & inputfile, std::map<std::string,double> &d_constants  )
{

  proc0cout << "\n Looking for constants in the header... " << std::endl;

  bool look = true;
  while ( look ){

    //    char ch = gzgetc( table_stream );
    char ch = table_stream.get();

    if ( ch == '#' ) {

      //  char key = gzgetc( table_stream );
      char key = table_stream.get() ;

      if ( key == 'K' ) {
        for (int i = 0; i < 3; i++ ){
          //  key = gzgetc( table_stream ); // reading the word KEY and space
          key = table_stream.get() ; // reading the word KEY and space
        }

        std::string name;
        while ( true ) {
          //  key = gzgetc( table_stream );
          key = table_stream.get();
          if ( key == '=' ) {
            break;
          }
          name.push_back( key );  // reading in the token's key name
        }

        std::string value_s;
        while ( true ) {
          //  key = gzgetc( table_stream );
          key = table_stream.get();
          if ( key == '\n' || key == '\t' || key == ' ' ) {
            break;
          }
          value_s.push_back( key ); // reading in the token's value
        }

        double value;
        sscanf( value_s.c_str(), "%lf", &value );

        proc0cout << " KEY found: " << name << " = " << value << std::endl;

        d_constants.insert( make_pair( name, value ) );

      } else {

        while ( true ) {
          // ch = gzgetc( table_stream ); // skipping this line
          ch = table_stream.get(); // skipping this line
          if ( ch == '\n' || ch == '\t' ) {
            break;
          }
        }
      }

    } else {

      look = false;

    }
  }
}
#endif

template <unsigned int max_dep_request_at_a_time>
static
Interp_class<max_dep_request_at_a_time>*
SCINEW_ClassicTable(std::string tableFileName, std::vector<std::string> requested_depVar_names={} ){
  // Create sub-ProblemSpecP object
  //
  // Obtain object parameters

  // This sets the table lookup variables and saves them in a map
  // Map<string name, Label>

  // READ TABLE:
  proc0cout << "--------------- Classic Arches Table Information---------------  " << std::endl;

  std::string uncomp_table_contents;

  int mpi_rank = Parallel::getMPIRank();

#ifndef OLD_TABLE
  char* table_contents=nullptr;
  int table_size = 0;

  if (mpi_rank == 0) {
    try {
      table_size = gzipInflate( tableFileName, uncomp_table_contents );
    }
    catch( Exception & e ) {
      throw ProblemSetupException( std::string("Call to gzipInflate() failed: ") + e.message(), __FILE__, __LINE__ );
    }

    table_contents = (char*) uncomp_table_contents.c_str();
    proc0cout << tableFileName << " is " << table_size << " bytes" << std::endl;
  }

  Uintah::MPI::Bcast(&table_size,1,MPI_INT,0,
      Parallel::getRootProcessorGroup()->getComm());

  if (mpi_rank != 0) {
    table_contents = scinew char[table_size];
  }

  Uintah::MPI::Bcast(table_contents, table_size, MPI_CHAR, 0,
      Parallel::getRootProcessorGroup()->getComm());

  std::stringstream table_contents_stream;
  table_contents_stream << table_contents;
#endif

  std::map<std::string, double>  d_constant;

#ifdef OLD_TABLE
  gzFile gzFp = gzopen(tableFileName.c_str(),"r");
  if( gzFp == nullptr ) {
    // If errno is 0, then not enough memory to uncompress file.
    proc0cout << "Error with gz in opening file: " << tableFileName << ". Errno: " << errno << "\n";
    throw ProblemSetupException("Unable to open the given input file: " + tableFileName, __FILE__, __LINE__);
  }

  checkForConstants(gzFp, tableFileName,d_constant );
  gzrewind(gzFp);
  Interp_class<max_dep_request_at_a_time>* return_pointer =  loadMixingTable<max_dep_request_at_a_time>(gzFp, tableFileName, requested_depVar_names ,d_constant );
  gzclose(gzFp);

  proc0cout << "Table successfully loaded into memory!" << std::endl;
  proc0cout << "---------------------------------------------------------------  " << std::endl;

  return return_pointer;
#else
 checkForConstants(table_contents_stream, tableFileName,d_constant );
  table_contents_stream.seekg(0);
 Interp_class<max_dep_request_at_a_time>* return_pointer = loadMixingTable<max_dep_request_at_a_time>(table_contents_stream,tableFileName,  requested_depVar_names  ,d_constant );
  if (mpi_rank != 0)
    delete [] table_contents;
  return return_pointer;
#endif

}
}
#endif
