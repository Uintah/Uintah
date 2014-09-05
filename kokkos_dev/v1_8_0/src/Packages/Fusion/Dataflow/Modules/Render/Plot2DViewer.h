/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

//    File   : Plot2DViewer.h
//    Author : Michael Callahan
//    Date   : April 2002

#if !defined(Plot2DViewer_h)
#define Plot2DViewer_h

#include <Dataflow/Network/Module.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Core/Datatypes/DenseMatrix.h>

#include <Core/GuiInterface/GuiVar.h>

#include <Packages/Fusion/share/share.h>

#include <Core/Datatypes/ScanlineField.h>
#include <Packages/Fusion/Core/Datatypes/StructHexVolField.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE Plot2DViewer : public Module {
public:
  Plot2DViewer(GuiContext* context);

  virtual ~Plot2DViewer();

  virtual void execute();

  void trueExecute( unsigned int port, unsigned int slice );

  virtual void tcl_command(GuiArgs&, void*);

private:
  GuiContext* ctx;

public:
  GuiInt havePLplot_;

private:
  GuiInt updateType_;

  GuiInt nPlots_;
  GuiInt nData_;

  GuiDouble xMin_;
  GuiDouble xMax_;
  GuiDouble yMin_;
  GuiDouble yMax_;
  GuiDouble zMin_;
  GuiDouble zMax_;

  bool updateGraph_;

  unsigned int ndata_;
  std::vector< unsigned int > idim_;
  std::vector< unsigned int > jdim_;
  std::vector< unsigned int > kdim_;

  std::vector< int > fGeneration_;
  std::vector< FieldHandle > fHandle_;

public:
  DenseMatrix * dMat_x_;
  DenseMatrix * dMat_y_;
  DenseMatrix * dMat_v_;
};

class Plot2DViewerAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src, int slice, Plot2DViewer* p2Dv) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *ftd,
				       const TypeDescription *ttd);
};


#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
class Plot2DViewerAlgoT : public Plot2DViewerAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle src, int slice, Plot2DViewer* p2Dv);
};


#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
void
Plot2DViewerAlgoT<FIELD, TYPE>::execute(FieldHandle field_h,
					int slice,
					Plot2DViewer* p2Dv )
{
  if( field_h.get_rep()->get_type_description(0)->get_name() == "StructHexVolField" ) {

    StructHexVolField<TYPE> *ifield =
      (StructHexVolField<TYPE> *) field_h.get_rep();

    typename StructHexVolField<TYPE>::mesh_handle_type imesh =
      ifield->get_typed_mesh();

    const unsigned int onx = imesh->get_nx();
    const unsigned int ony = imesh->get_ny();
    const unsigned int onz = imesh->get_nz();

    p2Dv->dMat_x_ = scinew DenseMatrix( onx, ony );
    p2Dv->dMat_y_ = scinew DenseMatrix( onx, ony );
    p2Dv->dMat_v_ = scinew DenseMatrix( onx, ony );

    typename StructHexVolField<TYPE>::mesh_type::Node::index_type node;
    typename StructHexVolField<TYPE>::value_type v;
    Point p;

    for (unsigned int i = 0; i < onx; i++) {
      for (unsigned int j = 0; j < ony; j++) {
	node.i_ = i;
	node.j_ = j;
	node.k_ = slice;

	ifield->value(v, node);
	p2Dv->dMat_v_->put( i, j, v );

	imesh->get_center(p, node);

	p2Dv->dMat_x_->put( i, j, sqrt(p.x() * p.x() + p.y() * p.y() ) );
	p2Dv->dMat_y_->put( i, j, p.z() );
      }
    }
  } 

  else if( field_h.get_rep()->get_type_description(0)->get_name() == "ScanlineField" ) {

    ScanlineField<TYPE> *ifield = (ScanlineField<TYPE> *) field_h.get_rep();

    typename ScanlineField<TYPE>::mesh_handle_type imesh = ifield->get_typed_mesh();
    const unsigned int onx = imesh->get_length();

    p2Dv->dMat_x_ = scinew DenseMatrix( onx, 1 );
    p2Dv->dMat_y_ = scinew DenseMatrix( onx, 1 );
    p2Dv->dMat_v_ = scinew DenseMatrix( onx, 1 );

    typename ScanlineField<TYPE>::mesh_type::Node::index_type node;
    typename ScanlineField<TYPE>::value_type v;

    Point p;

    for (unsigned int i = 0; i < onx; i++) {
      node = i;

      ifield->value(v, node);
      p2Dv->dMat_v_->put( i, 0, v );
      
      imesh->get_center(p, node);
      
      p2Dv->dMat_x_->put( i, 0, p.x() );
      p2Dv->dMat_y_->put( i, 0, v );
    }
  }
}

} // end namespace SCIRun

#endif // Plot2DViewer_h
