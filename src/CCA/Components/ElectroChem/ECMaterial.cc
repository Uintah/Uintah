/*
 * The MIT License
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


#include <CCA/Components/ElectroChem/ECMaterial.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/Patch.h>

#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <list>
#include <vector>

using namespace Uintah;

namespace ElectroChem {
  ECMaterial::ECMaterial( ProblemSpecP& ps, MaterialManagerP& mat_manager ) {
    std::list<GeometryObject::DataItem> geom_obj_data;

    geom_obj_data.push_back(GeometryObject::DataItem("concentration",
                                                      GeometryObject::Double));

    ps->require("diffusion_coeff", d_diff_coeff);

    for ( ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
          geom_obj_ps != nullptr;
          geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

      std::vector<GeometryPieceP> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPieceP mainpiece;
      if(pieces.size() == 0){
        throw ParameterNotFound("No piece specified in geom_object",
                               __FILE__, __LINE__);
      }else if(pieces.size() > 1){
        mainpiece = scinew UnionGeometryPiece(pieces);
      }else{
        mainpiece = pieces[0];
      }

      d_geom_objs.push_back(scinew GeometryObject(mainpiece,
                                                  geom_obj_ps, geom_obj_data));
    }
  }

  ECMaterial::~ECMaterial() {
    for (int i = 0; i< (int)d_geom_objs.size(); i++) {
      delete d_geom_objs[i];
    }
  }
} // End namespace ElectroChem
