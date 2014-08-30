/**
 * \file ExprPatch.h
 * \author James C. Sutherland
 *
 * Copyright (c) 2011 The University of Utah
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

#ifndef Expr_ExprPatch_h
#define Expr_ExprPatch_h

//--- stlib includes ---//
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <iomanip>


//--- SpatialOps Includes ---//
#include <spatialops/OperatorDatabase.h>


namespace Expr{

  class FieldManagerList;  // forward declaration
  struct FieldAllocInfo;

  //==================================================================



  /**
   *  @class  ExprPatch
   *  @author James C. Sutherland
   *  @date   May, 2007
   *
   *  This class currently only provides the most basic elements
   *  required for the ExpressionTree and ExpressionFactory classes.
   *
   *  It is provided for stand-alone code development in lieu of
   *  development within a framework environment.
   */
  class ExprPatch
  {
  public:

    /**
     *  @param nx Number of points in the x-direction.
     *  @param ny Number of points in the y-direction.
     *  @param nz Number of points in the z-direction.
     *  @param nparticle Number of particles on the patch
     *  @param nrawpts Number of points in pointwise fields (PointField types)
     */
    ExprPatch( const int nx, const int ny=1, const int nz=1,
               const size_t nparticle=0,
               const size_t nrawpts=0 );

    ~ExprPatch();

    inline int id() const{ return id_; }  // must define this method.

    FieldManagerList& field_manager_list();
    const FieldManagerList& field_manager_list() const;

    const SpatialOps::OperatorDatabase& operator_database() const{ return opDB_; }
    SpatialOps::OperatorDatabase& operator_database(){ return opDB_; }

    /**
     *  @brief Return the number of points (volumes) on this patch.
     */
    const std::vector<int>& dim() const{ return dims_; }

    /**
     *  Provided for compatibility with the default FieldManager.
     *  This allows the opportunity to dimension different field types
     *  differently.  An example application is AME, where we have
     *  field "bundles" that may have different number of points
     *  depending on the orientation of the bundle.
     */
    template<typename FieldT>
    const std::vector<int>& dim() const{ return dim(); }

    bool has_physical_bc_xplus() const{ return dims_[0]>1; }
    bool has_physical_bc_yplus() const{ return dims_[1]>1; }
    bool has_physical_bc_zplus() const{ return dims_[2]>1; }

    size_t get_n_particles() const{ return nparticles_; }

    size_t get_n_rawpoints() const{ return nrawpoints_; }

    FieldAllocInfo field_info() const;

  private:

    static int get_patch_id();

    const int id_;
    const size_t nparticles_, nrawpoints_;
    std::vector<int> dims_;

    FieldManagerList* const fieldMgrList_;
    SpatialOps::OperatorDatabase opDB_;

    ExprPatch( const ExprPatch& ); // no copying
  };


} // namespace Expr


#endif
