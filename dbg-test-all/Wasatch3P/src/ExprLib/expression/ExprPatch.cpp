/*
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

#include <expression/ExprPatch.h>
#include <expression/FieldManagerList.h>
#include <expression/FieldManager.h>

namespace Expr{

//--------------------------------------------------------------------

ExprPatch::ExprPatch( const int nx,
                      const int ny,
                      const int nz,
                      const size_t nparticle,
                      const size_t nrawpts )
  : id_( get_patch_id() ),
    nparticles_( nparticle ),
    nrawpoints_( nrawpts ),
    fieldMgrList_( new FieldManagerList("ExprPatch FM List") )
{
  dims_.assign(3,1);
  dims_[0]=nx; dims_[1]=ny; dims_[2]=nz;
}

//--------------------------------------------------------------------

ExprPatch::~ExprPatch()
{
  delete fieldMgrList_;
}

//--------------------------------------------------------------------

FieldManagerList&
ExprPatch::field_manager_list()
{
  return *fieldMgrList_;
}
//--------------------------------------------------------------------

const FieldManagerList&
ExprPatch::field_manager_list() const
{
  return *fieldMgrList_;
}

//--------------------------------------------------------------------

int
ExprPatch::get_patch_id()
{
  static int counter = 0;
  return ++counter;
}

//--------------------------------------------------------------------

FieldAllocInfo
ExprPatch::field_info() const
{
  return FieldAllocInfo( dim(),
                         get_n_particles(),
                         get_n_rawpoints(),
                         has_physical_bc_xplus(),
                         has_physical_bc_yplus(),
                         has_physical_bc_zplus() );
}

//--------------------------------------------------------------------

} // namespace Expr
