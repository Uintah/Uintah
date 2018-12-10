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


#include <CCA/Components/FVM/FVMMaterial.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/Patch.h>

#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <list>
#include <vector>

using namespace Uintah;

FVMMaterial::FVMMaterial( ProblemSpecP& ps, MaterialManagerP& materialManager,
                          FVMMethod method_type )
{
  d_method_type = method_type;
  std::list<GeometryObject::DataItem> geom_obj_data;

  if(d_method_type == ESPotential){
      geom_obj_data.push_back(GeometryObject::DataItem("conductivity",GeometryObject::Double));
  }

  if(d_method_type == Gauss){
    geom_obj_data.push_back(GeometryObject::DataItem("pos_charge_density",GeometryObject::Double));
    geom_obj_data.push_back(GeometryObject::DataItem("neg_charge_density",GeometryObject::Double));
    geom_obj_data.push_back(GeometryObject::DataItem("permittivity",GeometryObject::Double));
  }

  for ( ProblemSpecP geom_obj_ps = ps->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

    std::vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPieceP mainpiece;
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
    }
    else if(pieces.size() > 1){
      mainpiece = scinew UnionGeometryPiece(pieces);
    }
    else {
      mainpiece = pieces[0];
    }

    d_geom_objs.push_back(scinew GeometryObject(mainpiece, geom_obj_ps, geom_obj_data));
  }
}

FVMMaterial::~FVMMaterial()
{
  for (int i = 0; i< (int)d_geom_objs.size(); i++) {
    delete d_geom_objs[i];
  }
}

void
FVMMaterial::initializeConductivity(CCVariable<double>& conductivity, const Patch* patch)
{
  conductivity.initialize(0.0);

  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
      GeometryPieceP piece = d_geom_objs[obj]->getPiece();

    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Point center = patch->cellPosition(c);

      if(piece->inside(center)){
        conductivity[c] = d_geom_objs[obj]->getInitialData_double("conductivity");
      }
    }
  }
}

void FVMMaterial::initializePermittivityAndCharge(CCVariable<double>& permittivity,
                                                  CCVariable<double>& pos_charge,
                                                  CCVariable<double>& neg_charge,
                                                  const Patch* patch)
{

  Vector dx = patch->dCell();
  double volume = dx.x() * dx.y() * dx.z();

  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
        GeometryPieceP piece = d_geom_objs[obj]->getPiece();

      for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        Point center = patch->cellPosition(c);

        if(piece->inside(center)){
          permittivity[c] += d_geom_objs[obj]->getInitialData_double("permittivity");
          pos_charge[c] += volume * d_geom_objs[obj]->getInitialData_double("pos_charge_density");
          neg_charge[c] += volume * d_geom_objs[obj]->getInitialData_double("neg_charge_density");
        }
      }
    }
}
