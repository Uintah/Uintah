/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef MixMdlHelper_h
#define MixMdlHelper_h

#include <string>

#include <tabprops/TabProps.h>

// forward declarations
class StateTable;
class PresumedPDFMixMdl;
class Integrator;

/**
 *  @class  MixMdlHelper
 *  @author James C. Sutherland
 *  @date   March, 2005
 *
 *  @brief Implements mixing models, given a reaction model.
 */
class MixMdlHelper{
public:

  /**
   *  @param mixingModel : Choice of the presumed-pdf mixing model to be used
   *  @param rxnMdl : The tabular form of the reaction model.
   *  @param convVarNam : The name of the variable to be convoluted.
   *  @param order the order of interpolant
   */
  MixMdlHelper( PresumedPDFMixMdl & mixingModel,
		StateTable & rxnMdl,
		const std::string & convVarNam,
		const int order );

  ~MixMdlHelper( );

  /**
   * Set the mesh for the given variable.  The mesh is assumed to be structured.  The extent
   * of the mesh in one dimension cannot be a function of any other dimension.
   */
  void set_mesh( const std::string & varname, const std::vector<double> & mesh );

  /**
   *  Set the mesh for all variables.
   */
  void set_mesh( const std::vector< std::vector<double> > & mesh );


  /**
   *  Specify the integrator to use.
   */
  void set_integrator( Integrator * i );

  /**
   *  Implement the mixing model.  This iterates each entry in the reaction model state
   *  table and applies the mixing model on the specified mesh.
   */
  void implement();

  /**
   *  Access function to return a handle to the reaction model.
   */
  StateTable & reaction_model(){ return rxnMdl_; }


private:

  MixMdlHelper( const MixMdlHelper& );  // no copying

  int findix( const std::string & varname ) const;

  void setup_mesh_point( const int ipoint,
			 std::vector<double> & values );

  void apply_on_mesh( const std::string&, const InterpT* const );

  void export_variable( const std::vector<double> & values,
			const std::string & name );

  void verify_interp( const std::vector<double> & values,
		      const InterpT* const interp,
		      const std::string & name );

  PresumedPDFMixMdl & mixMdl_;
  StateTable & rxnMdl_;
  const int nDimRxn_, nDim_, order_;
  StateTable * mixMdlTable_;

  std::vector<int> npts_;
  std::vector<std::string> indepVarNames_;

  std::vector< std::vector<double> > mesh_;
  std::vector<double> outputVarVals_;

  double loBound_, hiBound_;

  int convVarIndex_;

};

#endif
