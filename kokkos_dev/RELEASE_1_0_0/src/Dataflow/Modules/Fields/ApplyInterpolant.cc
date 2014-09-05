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

/*
 *  ApplyInterpolant.cc:  Apply an interpolant field to project the data
 *                 from one field onto the mesh of another field.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/Dispatch2.h>
#include <Dataflow/Modules/Fields/InterpolantTypes.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::cerr;
using std::vector;
using std::pair;


class ApplyInterpolant : public Module {
public:
  ApplyInterpolant(const clString& id);
  virtual ~ApplyInterpolant();

  virtual void execute();

  template <class FSRC, class FITP, class FRES>
  void callback(FSRC *fsrc, FITP *fitp, FRES *);
};


extern "C" Module* make_ApplyInterpolant(const clString& id)
{
  return new ApplyInterpolant(id);
}


ApplyInterpolant::ApplyInterpolant(const clString& id)
  : Module("ApplyInterpolant", id, Filter, "Fields", "SCIRun")
{
}


ApplyInterpolant::~ApplyInterpolant()
{
}



template <class FSRC, class FITP, class FOUT>
void
ApplyInterpolant::callback(FSRC *fsrc, FITP *fitp, FOUT *)
{
  FOUT *fout = new FOUT(fitp->get_typed_mesh(), fitp->data_at());
  FieldHandle fhout(fout);

  switch(fout->data_at())
  {
  case Field::NODE:
    {
      typename FOUT::mesh_type::node_iterator iter =
	fout->get_typed_mesh()->node_begin();
      while (iter != fout->get_typed_mesh()->node_end())
      {
	typename FITP::value_type v;
	fitp->value(v, *iter);
	if (!v.empty())
	{
	  typename FSRC::value_type val =
	    (typename FSRC::value_type)(fsrc->value(v[0].first) * v[0].second);
	  unsigned int j;
	  for (j = 1; j < v.size(); j++)
	  {
	    val += (typename FSRC::value_type)
	      (fsrc->value(v[j].first) * v[j].second);
	  }
	  fout->set_value(val, *iter);
	}
	++iter;
      }
    }
    break;

  case Field::EDGE:
    {
      typename FOUT::mesh_type::edge_iterator iter =
	fout->get_typed_mesh()->edge_begin();
      while (iter != fout->get_typed_mesh()->edge_end())
      {
	typename FITP::value_type v;
	fitp->value(v, *iter);
	if (!v.empty())
	{
	  typename FSRC::value_type val =
	    (typename FSRC::value_type)(fsrc->value(v[0].first) * v[0].second);
	  unsigned int j;
	  for (j = 1; j < v.size(); j++)
	  {
	    val += (typename FSRC::value_type)
	      (fsrc->value(v[j].first) * v[j].second);
	  }
	  fout->set_value(val, *iter);
	}
	++iter;
      }
    }
    break;

  case Field::FACE:
    {
      typename FOUT::mesh_type::face_iterator iter =
	fout->get_typed_mesh()->face_begin();
      while (iter != fout->get_typed_mesh()->face_end())
      {
	typename FITP::value_type v;
	fitp->value(v, *iter);
	if (!v.empty())
	{
	  typename FSRC::value_type val =
	    (typename FSRC::value_type)(fsrc->value(v[0].first) * v[0].second);
	  unsigned int j;
	  for (j = 1; j < v.size(); j++)
	  {
	    val += (typename FSRC::value_type)
	      (fsrc->value(v[j].first) * v[j].second);
	  }
	  fout->set_value(val, *iter);
	}
	++iter;
      }
    }
    break;

  case Field::CELL:
    {
      typename FOUT::mesh_type::cell_iterator iter =
	fout->get_typed_mesh()->cell_begin();
      while (iter != fout->get_typed_mesh()->cell_end())
      {
	typename FITP::value_type v;
	fitp->value(v, *iter);
	if (!v.empty())
	{
	  typename FSRC::value_type val =
	    (typename FSRC::value_type)(fsrc->value(v[0].first) * v[0].second);
	  unsigned int j;
	  for (j = 1; j < v.size(); j++)
	  {
	    val += (typename FSRC::value_type)
	      (fsrc->value(v[j].first) * v[j].second);
	  }
	  fout->set_value(val, *iter);
	}
	++iter;
      }
    }
    break;

  default:
    cout << "No data in interpolant field.";
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output");
  ofp->send(fhout);
}



#define HAIRY_MACRO(FSRC, FITP, DSRC)\
switch(src_field->data_at())\
{\
case Field::NODE:\
  {\
	FSRC<DSRC> *src = dynamic_cast<FSRC<DSRC> *>(src_field);\
	FITP<vector<pair<FSRC<DSRC>::mesh_type::node_index, double> > > *itp =\
	  dynamic_cast<FITP<vector<pair<FSRC<DSRC>::mesh_type::node_index, double> > > *>(itp_field);\
	if (src && itp)\
	{\
	  callback(src, itp, (FITP<DSRC> *)0);\
	}\
	else\
	{\
	  cout << "Incorrect field types dispatched\n";\
	}\
  }\
  break;\
\
case Field::EDGE:\
  {\
	FSRC<DSRC> *src = dynamic_cast<FSRC<DSRC> *>(src_field);\
	FITP<vector<pair<FSRC<DSRC>::mesh_type::edge_index, double> > > *itp =\
	  dynamic_cast<FITP<vector<pair<FSRC<DSRC>::mesh_type::edge_index, double> > > *>(itp_field);\
	if (src && itp)\
	{\
	  callback(src, itp, (FITP<DSRC> *)0);\
	}\
	else\
	{\
	  cout << "Incorrect field types dispatched\n";\
	}\
  }\
  break;\
\
case Field::FACE:\
  {\
	FSRC<DSRC> *src = dynamic_cast<FSRC<DSRC> *>(src_field);\
	FITP<vector<pair<FSRC<DSRC>::mesh_type::face_index, double> > > *itp =\
	  dynamic_cast<FITP<vector<pair<FSRC<DSRC>::mesh_type::face_index, double> > > *>(itp_field);\
	if (src && itp)\
	{\
	  callback(src, itp, (FITP<DSRC> *)0);\
	}\
	else\
	{\
	  cout << "Incorrect field types dispatched\n";\
	}\
  }\
  break;\
\
case Field::CELL:\
  {\
	FSRC<DSRC> *src = dynamic_cast<FSRC<DSRC> *>(src_field);\
	FITP<vector<pair<FSRC<DSRC>::mesh_type::cell_index, double> > > *itp =\
	  dynamic_cast<FITP<vector<pair<FSRC<DSRC>::mesh_type::cell_index, double> > > *>(itp_field);\
	if (src && itp)\
	{\
	  callback(src, itp, (FITP<DSRC> *)0);\
	}\
	else\
	{\
	  cout << "Incorrect field types dispatched\n";\
	}\
  }\
  break;\
\
default:\
  cout << "No data location to dispatch on.\n";\
  return;\
}


void
ApplyInterpolant::execute()
{
  FieldIPort *src_port = (FieldIPort *)get_iport("Source");
  FieldHandle sfieldhandle;
  Field *src_field;
  if (!(src_port->get(sfieldhandle) && (src_field = sfieldhandle.get_rep())))
  {
    return;
  }

  FieldIPort *itp_port = (FieldIPort *)get_iport("Interpolant");
  FieldHandle ifieldhandle;
  Field *itp_field;
  if (!(itp_port->get(ifieldhandle) && (itp_field = ifieldhandle.get_rep())))
  {
    return;
  }

  const string src_geom_name = src_field->get_type_name(0);
  const string src_data_name = src_field->get_type_name(1);
  const string itp_geom_name = itp_field->get_type_name(0);

  cout << "src geom " << src_geom_name << '\n';
  cout << "src data " << src_data_name << '\n';
  cout << "itp geom " << itp_geom_name << '\n';
  cout << "itp type " << itp_field->get_type_name() << '\n';

  if (src_geom_name == "TetVol")
  {
#if 0
    if (itp_geom_name == "TetVol")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(TetVol, TetVol, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(TetVol, TetVol, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(TetVol, TetVol, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(TetVol, TetVol, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(TetVol, TetVol, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(TetVol, TetVol, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(TetVol, TetVol, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(TetVol, TetVol, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(TetVol, TetVol, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else if (itp_geom_name == "LatticeVol")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(TetVol, LatticeVol, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(TetVol, LatticeVol, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(TetVol, LatticeVol, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(TetVol, LatticeVol, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(TetVol, LatticeVol, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(TetVol, LatticeVol, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(TetVol, LatticeVol, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(TetVol, LatticeVol, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(TetVol, LatticeVol, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
#endif
    if (itp_geom_name == "TriSurf")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(TetVol, TriSurf, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(TetVol, TriSurf, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(TetVol, TriSurf, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(TetVol, TriSurf, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(TetVol, TriSurf, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(TetVol, TriSurf, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(TetVol, TriSurf, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(TetVol, TriSurf, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(TetVol, TriSurf, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else if (itp_geom_name == "ContourField")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(TetVol, ContourField, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(TetVol, ContourField, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(TetVol, ContourField, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(TetVol, ContourField, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(TetVol, ContourField, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(TetVol, ContourField, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(TetVol, ContourField, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(TetVol, ContourField, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(TetVol, ContourField, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else if (itp_geom_name == "PointCloud")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(TetVol, PointCloud, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(TetVol, PointCloud, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(TetVol, PointCloud, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(TetVol, PointCloud, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(TetVol, PointCloud, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(TetVol, PointCloud, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(TetVol, PointCloud, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(TetVol, PointCloud, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(TetVol, PointCloud, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else
    {
      cout << "bad interp field type";
    }
  }
#if 0
  else if (src_geom_name == "LatticeVol")
  {
    if (itp_geom_name == "TetVol")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(LatticeVol, TetVol, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(LatticeVol, TetVol, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(LatticeVol, TetVol, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(LatticeVol, TetVol, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(LatticeVol, TetVol, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(LatticeVol, TetVol, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(LatticeVol, TetVol, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(LatticeVol, TetVol, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(LatticeVol, TetVol, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else if (itp_geom_name == "LatticeVol")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(LatticeVol, LatticeVol, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else if (itp_geom_name == "TriSurf")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(LatticeVol, TriSurf, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else if (itp_geom_name == "ContourField")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(LatticeVol, ContourField, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(LatticeVol, ContourField, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(LatticeVol, ContourField, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(LatticeVol, ContourField, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(LatticeVol, ContourField, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(LatticeVol, ContourField, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(LatticeVol, ContourField, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(LatticeVol, ContourField, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(LatticeVol, ContourField, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else if (itp_geom_name == "PointCloud")
    {
      if (src_data_name == "Vector")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, Vector);
      }
      else if (src_data_name == "double")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, double);
      }
      else if (src_data_name == "float")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, float);
      }
      else if (src_data_name == "int")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, int);
      }
      else if (src_data_name == "unsigned int")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, unsigned int);
      }
      else if (src_data_name == "short")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, short);
      }
      else if (src_data_name == "unsigned short")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, unsigned short);
      }
      else if (src_data_name == "char")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, char);
      }
      else if (src_data_name == "unsigned char")
      {
	HAIRY_MACRO(LatticeVol, PointCloud, unsigned char);
      }
      else
      {
	cout << "Non-interpable source field type\n";
      }
    }
    else
    {
      cout << "bad interp field type";
    }
  }
#endif
  else
  {
    cout << "bad source field type";
  }
}

} // End namespace SCIRun


