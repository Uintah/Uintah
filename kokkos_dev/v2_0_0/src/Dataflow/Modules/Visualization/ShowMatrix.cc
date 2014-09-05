/*
 *  ShowMatrix.cc:
 *
 *  Written by:
 *   McKay Davis
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   June 2003
 *  Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomQuads.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/Geom/GeomDL.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomGrid.h>
#include <Core/Geom/GeomText.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>


#include <stdio.h>
#include <iostream>
#include <map>

namespace SCIRun {

class ShowMatrix : public Module {
private:
  
  // These are the NORMAL vectors for said surface orientation
  const Vector right, left, up, down, front, back;

  Point			origin_;
  MaterialHandle	white_;
  MaterialHandle	black_;
  ColorMapHandle	cmap_;
  int			cgen_;
  bool			swap_row_col_;
  GeomHandle		plot_;

  int			cached_gui_gmode_;
  int			cached_gui_showtext_;
  int			cached_gui_cmode_;
  
  GuiInt		gui_color_by_val_;
  GuiInt		gui_grid_x_;
  GuiInt		gui_grid_y_;
  GuiInt		gui_grid_z_;
  GuiDouble		gui_trans_x_;
  GuiDouble		gui_trans_y_;
  GuiDouble		gui_scale_;
  GuiDouble		gui_scale_x_;
  GuiDouble		gui_scale_y_;
  GuiString		gui_mode_;
  GuiInt		gui_gmode_;
  GuiInt		gui_showtext_;
  GuiInt		gui_row_begin_;
  GuiInt		gui_row_end_;
  GuiInt		gui_col_begin_;
  GuiInt		gui_col_end_;
  GuiDouble		gui_x_gap_;
  GuiDouble		gui_z_gap_;
  GuiInt		gui_data_face_centered_;
  GuiInt		gui_cmode_;
  
  struct MatrixData {
    double	min;
    double	max;
    int		row_begin;
    int		row_end;
    int		col_begin;
    int		col_end;
    int		generation;
    bool	changed;
  };
  
  enum color_mode_t {
    COLOR_BY_ROW,
    COLOR_BY_COL,
    COLOR_BY_VAL
  };
  color_mode_t				color_mode_;
  
  map<Matrix*, MatrixData>		matrix_data_;
  const MatrixData &			get_matrix_data(MatrixHandle);

  double				get_value(MatrixHandle,int,int);
  MaterialHandle			get_color(MatrixHandle,int,int);
  void					set_color_scale(MatrixHandle);						      

  
  GeomHandle	generate_grid(bool do_3d=true);
  GeomHandle	generate_text(MatrixHandle);
  GeomHandle	generate_line_graph(MatrixHandle);
  GeomHandle	generate_3d_sheet_graph(MatrixHandle);
  GeomHandle	generate_3d_bar_graph(MatrixHandle);
  GeomHandle	generate_3d_ribbon_graph(MatrixHandle, bool sides=false);
  GeomHandle	generate_contour(MatrixHandle);

public:
  ShowMatrix(GuiContext*);
  virtual ~ShowMatrix();
  virtual void execute();
};

  DECLARE_MAKER(ShowMatrix)

ShowMatrix::ShowMatrix(GuiContext* ctx)
  : Module("ShowMatrix", ctx, Filter, "Visualization", "SCIRun"),
    right(1.0,0.0,0.0), left(-1.0,0.0,0.0), 
    up(0.0,1.0,0.0), down(0.0,-1.0,0.0),
    front(0.0,0.0,1.0), back(0.0,0.0,-1.0),
    white_(scinew Material(Color(0,0,0), Color(1,1,1), Color(1,1,1), 20)),
    black_(scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 20)),
    swap_row_col_(false),
    gui_color_by_val_(ctx->subVar("color_by_val")),
    gui_grid_x_(ctx->subVar("grid_x")),
    gui_grid_y_(ctx->subVar("grid_y")),
    gui_grid_z_(ctx->subVar("grid_z")),
    gui_trans_x_(ctx->subVar("xpos")),
    gui_trans_y_(ctx->subVar("ypos")),
    gui_scale_(ctx->subVar("xscale")),
    gui_scale_x_(ctx->subVar("xscale")),
    gui_scale_y_(ctx->subVar("yscale")),
    gui_mode_(ctx->subVar("displaymode")),
    gui_gmode_(ctx->subVar("gmode")),
    gui_showtext_(ctx->subVar("showtext")),
    gui_row_begin_(ctx->subVar("row_begin")),
    gui_row_end_(ctx->subVar("row_end")),
    gui_col_begin_(ctx->subVar("col_begin")),
    gui_col_end_(ctx->subVar("col_end")),
    gui_x_gap_(ctx->subVar("xgap")),
    gui_z_gap_(ctx->subVar("ygap")),
    gui_data_face_centered_(ctx->subVar("data_face_centered")),
    gui_cmode_(ctx->subVar("colormapmode"))
{

}


ShowMatrix::~ShowMatrix()
{
}


// This method exists solely to give the user the option
// to swap the row/column representation of the Matrix.
// (And a little sanity check)
// This could be expanded to handle SparseMatrix later
double
ShowMatrix::get_value(MatrixHandle mh, int row, int col)
{
  if (swap_row_col_) {
    const int temp = row;
    row = col;
    col = temp;
  }
  ASSERT(row >= 0);
  ASSERT(row < mh->nrows());
  ASSERT(col >= 0);
  ASSERT(col < mh->ncols());
  
  return mh->get(row, col);
}

MaterialHandle
ShowMatrix::get_color(MatrixHandle mh, int row, int col)
{
  const MatrixData &data = get_matrix_data(mh);
  switch (color_mode_)
    {
    case COLOR_BY_ROW: return cmap_->lookup((row-data.row_begin)/double(data.row_end-data.row_begin));break;
    case COLOR_BY_COL: return cmap_->lookup((col-data.col_begin)/double(data.col_end-data.col_begin));break;
    default:
    case COLOR_BY_VAL: return cmap_->lookup(get_value(mh,row,col));break;
    }
}

void
ShowMatrix::set_color_scale(MatrixHandle mh)
{
  switch (color_mode_)
    {
    case COLOR_BY_ROW: 
    case COLOR_BY_COL: cmap_->Scale(0.0, 1.0); break;
    default:
    case COLOR_BY_VAL: cmap_->Scale(get_matrix_data(mh).min,
				    get_matrix_data(mh).max); break;
    }  
}
  
// Generates a very simple half cube grid with X by Y by Z divisions
GeomHandle
ShowMatrix::generate_grid(bool do_3d)
{ 
  gui_grid_x_.set(10);
  gui_grid_y_.set(10);
  gui_grid_z_.set(10);

  const int gridx = gui_grid_x_.get();
  const int gridy = gui_grid_y_.get();
  const int gridz = gui_grid_z_.get();
  
  GeomLines *grid = scinew GeomLines();
  int i;
  for (i = 0; i <= gridx; i++) {
    grid->add(Point(i,0,0), Point(i,gridy,0));
    if (do_3d) grid->add(Point(i,0,0), Point(i,0,gridz));
  }
  
  for (i = 0; i <= gridy; i++) {
    grid->add(Point(0,i,0), Point(gridx,i,0));
    if (do_3d) grid->add(Point(0,i,0), Point(0,i,gridz));
  }
    
  if (do_3d) for (i = 0; i <= gridz; i++) {
    grid->add(Point(0,0,i), Point(gridx,0,i));
    grid->add(Point(0,0,i), Point(0,gridy,i));  
  }
  GeomTransform *graph = scinew GeomTransform(grid);
  graph->scale(Vector(1.0/gridx,1.0/gridy,1.0/gridz));

  return graph;
}

GeomHandle
ShowMatrix::generate_text(MatrixHandle mh)
{
  ostringstream buffer;
  buffer.precision(2);

  GeomTexts *text = scinew GeomTexts;
  text->set_font_index(2);
  const MatrixData &data = get_matrix_data(mh);
  for(int row = data.row_begin; row <= data.row_end; row++) {
    for (int col = data.col_begin; col <= data.col_end; col++) {
      buffer.str("");
      buffer << get_value(mh,row,col);
      text->add(buffer.str(), Point(col, get_value(mh, row, col), row));
    }
  }
  GeomTransform *graph = scinew GeomTransform(text);
  switch (gui_gmode_.get()){
  case 1:
  case 3:
    graph->translate(Vector(-data.col_begin, -data.min, -data.row_begin));
    graph->scale(Vector(1.0 / (data.col_end - data.col_begin), 
			1.0 / (data.max - data.min),
			1.0 / (data.row_end - data.row_begin)));
    
    break;
  case 2:
    graph->translate(Vector(-data.col_begin+0.5, -data.min, -data.row_begin+0.5));
    graph->scale(Vector(1.0 / (data.col_end - data.col_begin + 1.0), 
			1.0 / (data.max - data.min),
			1.0 / (data.row_end - data.row_begin + 1.0)));
	break;
	
  default:
  case 4:
  case 5:
    graph->translate(Vector(-data.col_begin, -data.min, -data.row_begin+0.5));
    graph->scale(Vector(1.0 / (data.col_end - data.col_begin), 
			1.0 / (data.max - data.min),
			1.0 / (data.row_end - data.row_begin + 1.0)));
    break;
  }
  return graph;
}


GeomHandle 
ShowMatrix::generate_line_graph(MatrixHandle mh)
{
  const MatrixData &data = get_matrix_data(mh);
  set_color_scale(mh);
  GeomCLineStrips *lines = scinew GeomCLineStrips();
  lines->setLineWidth(3.0);
  for(int row = data.row_begin; row <= data.row_end; row++) {
    lines->newline();
    for (int col = data.col_begin; col <= data.col_end; col++) {
      lines->add(Point(col, get_value(mh, row, col), row), get_color(mh,row,col));
    }
  }
  GeomTransform *graph = scinew GeomTransform(lines);
  if (gui_data_face_centered_.get() == 1) {
  graph->translate(Vector(-data.col_begin + 0.5, -data.min, -data.row_begin + 0.5));
  graph->scale(Vector(1.0 / (data.col_end - data.col_begin + 1.0), 
		      1.0 / (data.max - data.min),
		      1.0 / (data.row_end - data.row_begin + 1.0)));
  }
  else {
    graph->translate(Vector(-data.col_begin, -data.min, -data.row_begin));
    graph->scale(Vector(1.0 / (data.col_end - data.col_begin), 
			1.0 / (data.max - data.min),
			1.0 / (data.row_end - data.row_begin)));
  }

  return graph;
}


GeomHandle 
ShowMatrix::generate_3d_ribbon_graph(MatrixHandle mh,bool fill_sides)
{  
  gui_x_gap_.set(0.2);
  gui_z_gap_.set(0.2);


  set_color_scale(mh);

  const MatrixData &data = get_matrix_data(mh);
  double midpoint = data.min;
  //double x_gap = gui_x_gap_.get();
  double z_gap = gui_z_gap_.get();
  GeomGroup *bars = scinew GeomGroup();

  for(int z = data.row_begin; z <= data.row_end; z++) {
    for (int x = data.col_begin; x < data.col_end; x++) {
      GeomFastQuads *bar = scinew GeomFastQuads();
      bars->add(bar);
      if (fill_sides){
	//front
	bar->add(Point(x, get_value(mh,z,x), z + 1.0 - z_gap), 
		 front, get_color(mh,z,x),
		 Point(x, midpoint, z + 1.0 - z_gap),
		 front, black_,
		 Point(x + 1, midpoint, z + 1.0 - z_gap),
		 front, black_,
		 Point(x + 1, get_value(mh,z,x+1), z + 1.0 - z_gap), 
		 front, get_color(mh,z,x+1));
	
	//back
	bar->add(Point(x + 1, get_value(mh,z,x+1), z + z_gap), 
		 back, get_color(mh,z,x+1),
		 Point(x + 1, midpoint, z + z_gap),
		 back, black_,
		 Point(x, midpoint, z + z_gap),
		 back, black_,
		 Point(x, get_value(mh,z,x), z +z_gap), 
		 back, get_color(mh,z,x));
      }

      Vector e1(0,0,1);
      Vector e2(1,get_value(mh,z,x+1)-get_value(mh,z,x),0);
      e2.normalize();
      Vector normal = Cross(e1,e2);
      normal.normalize();

      bar->add(Point(x, get_value(mh,z,x), z),
	       normal, get_color(mh,z,x),
	       Point(x, get_value(mh,z,x), z+1),
	       normal, get_color(mh,z,x),
	       Point(x+1, get_value(mh,z,x+1), z+1),
	       normal, get_color(mh,z,x+1),
	       Point(x+1, get_value(mh,z,x+1), z),
	       normal, get_color(mh,z,x+1));
	
#if 0
      // top
      // Note:: The normal vectors for this top just point straight up (wrong)
      bar->add(Point(x * dx, get_value(mh,z,x) * dy + y_off, z * dz + z_off),
	       Vector(0.0, 1.0, 0.0),
	       get_color(mh,z,x),
	       Point(x * dx, get_value(mh,z,x) * dy + y_off, z * dz + dz - z_off),
	       Vector(0.0, 1.0, 0.0),
	       get_color(mh,z,x),
	       Point(x * dx + dx, get_value(mh,z,x+1) * dy + y_off, z * dz + dz - z_off),
	       Vector(0.0, 1.0, 0.0),
	       get_color(mh,z,x+1),
	       Point(x * dx + dx, get_value(mh,z,x+1) * dy + y_off, z * dz + z_off),
	       Vector(0.0, 1.0, 0.0),
	       get_color(mh,z,x+1));
#endif
    }
  }
  GeomTransform *graph = scinew GeomTransform(bars);
  graph->translate(Vector(-data.col_begin, -data.min, -data.row_begin));
  graph->scale(Vector(1.0 / (data.col_end - data.col_begin), 
		      1.0 / (data.max - data.min),
		      1.0 / (data.row_end - data.row_begin + 1.0)));
  return graph;
}


GeomHandle 
ShowMatrix::generate_3d_sheet_graph(MatrixHandle mh)
{
  const MatrixData &data = get_matrix_data(mh);  
  set_color_scale(mh);
  int x,z;
  GeomGroup *bars = scinew GeomGroup();
  for(z = data.row_begin; z < data.row_end; z++) {
    for (x = data.col_begin; x < data.col_end; x++) {
      GeomFastQuads *bar = scinew GeomFastQuads();
      bars->add(bar);
      Point p0(x, get_value(mh,z,x), z);
      Point p1(x, get_value(mh,z+1,x), z+1);
      Point p2(x+1, get_value(mh,z+1,x+1), z+1);
      Point p3(x+1, get_value(mh,z,x+1), z);
      Vector n1(Cross((p1-p0),(p3-p0)));
      Vector n2(Cross((p2-p1),(p0-p1)));
      Vector n3(Cross((p3-p2),(p1-p2)));
      Vector n4(Cross((p0-p3),(p2-p3)));
      n1.normalize();
      n2.normalize();
      n3.normalize();
      n4.normalize();

      Vector normal = (n1 + n2 + n3 + n4);
      normal.normalize();

      bar->add(p0, normal, get_color(mh,z,x),
	       p1, normal, get_color(mh,z+1,x),
	       p2, normal, get_color(mh,z+1,x+1),
	       p3, normal, get_color(mh,z,x+1));
    }
  }
  GeomTransform *sheet = scinew GeomTransform(bars);
  sheet->translate(Vector(-data.col_begin, -data.min, -data.row_begin));
  sheet->scale(Vector(1.0 / (data.col_end - data.col_begin), 
		      1.0 / (data.max - data.min),
		      1.0 / (data.row_end - data.row_begin)));
		      

  cmap_->ResetScale();
  return sheet;
}


GeomHandle 
ShowMatrix::generate_contour(MatrixHandle mh)
{
  const MatrixData &data = get_matrix_data(mh);
  
  //const double dy = 1.0 / (data.max - data.min);
  const double dx = 1.0 / (data.col_end - data.col_begin);
  const double dz = 1.0 / (data.row_end - data.row_begin);

  int row, col;
  
  const double x_off = 0.0;
  const double z_off = 0.0;

  Vector normal(0.0,1.0,0.0);
  cmap_->Scale(data.min,data.max);
  GeomFastQuads *quad = scinew GeomFastQuads();
  for (row = data.row_begin; row < data.row_end; row++) {
    for (col = data.col_begin; col < data.col_end; col++) {
      quad->add(Point(col * dx + x_off, 0.0, row * dz + z_off),
	       normal,
	       cmap_->lookup(mh->get(row,col)),
	       Point(col * dx + x_off, 0.0, row * dz + z_off + dz),
	       normal,
	       cmap_->lookup(mh->get(row+1,col)),
	       Point(col * dx + x_off + dx, 0.0, row * dz + z_off + dz),
	       normal,
	       cmap_->lookup(mh->get(row+1,col+1)),
	       Point(col * dx + x_off + dx, 0.0, row * dz + z_off),
	       normal,
	       cmap_->lookup(mh->get(row,col+1)));
    }
  }
  cmap_->ResetScale();
  return quad;
}



GeomHandle 
ShowMatrix::generate_3d_bar_graph(MatrixHandle mh)
{
  double x_gap = gui_x_gap_.get();
  double z_gap = gui_z_gap_.get();

  set_color_scale(mh);

  const MatrixData &data = get_matrix_data(mh);
  double midpoint = data.min;
  GeomGroup *bars= scinew GeomGroup();
  for(int z = data.row_begin; z <= data.row_end; z++) {
    for (int x = data.col_begin; x <= data.col_end; x++) {
      const double y = get_value(mh,z,x);
      // If this bar is essentially flat, dont render it at all
      if (fabs(y - midpoint) < ((data.max - data.min)/200.0)) continue;
      
      MaterialHandle color = get_color(mh,z,x);
      GeomFastQuads *bar = scinew GeomFastQuads();
      bars->add(bar);

      // Corners of box. Naming convention goes:
      // 1st letter: [f]ront or [b]ack
      // 2nd letter: [u]pper or [l]ower
      // 3rd letter: [l]eft  or [r]ight
      const Point flr(x + 1 - x_gap, midpoint, z + 1 - z_gap);
      const Point fur(x + 1 - x_gap, y, z + 1 - z_gap);
      const Point ful(x + x_gap, y, z + 1 - z_gap);
      const Point fll(x + x_gap, midpoint, z + 1 - z_gap);
      const Point blr(x + 1 - x_gap, midpoint, z + z_gap);
      const Point bur(x + 1 - x_gap, y, z + z_gap);
      const Point bul(x + x_gap, y, z + z_gap);
      const Point bll(x + x_gap, midpoint, z + z_gap);

      //front quad
      bar->add(flr, front, black_, fur, front, color,
	       ful, front, color, fll, front, black_);

      //back quad
      bar->add(blr, back, black_, bll, back, black_,
	       bul, back, color, bur, back, color);
      
      // left quad
      bar->add(bul, left, color, bll, left, black_,
	       fll, left, black_, ful, left, color);

      // right quad
      bar->add(fur, right, color, flr, right, black_,
	       blr, right, black_, bur, right, color);
      // top quad
      bar->add(bul,up,color,ful,up,color,
	       fur,up,color,bur,up,color);

      // bottom quad
      bar->add(fll,down,black_,bll,down,black_,
	       blr,down,black_,flr,down,black_);
    }
  }
  GeomTransform *graph = scinew GeomTransform(bars);
  graph->translate(Vector(-data.col_begin, -data.min, -data.row_begin));
  graph->scale(Vector(1.0 / (data.col_end - data.col_begin + 1), 
		      1.0 / (data.max - data.min),
		      1.0 / (data.row_end - data.row_begin + 1)));
  
  return graph;
}






void
ShowMatrix::execute()
{
  //int deb = 0;
  vector<MatrixHandle> matrices;
  port_range_type range = get_iports("Matrix");
  if (range.first == range.second) {
    return;
  }

  port_map_type::iterator pi = range.first;
  while (pi != range.second) {
    MatrixIPort *iport = (MatrixIPort *)get_iport(pi->second);
    pi++;
    if (!iport) {
      error("Unable to initialize iport '");// + to_string(pi->second) + "'.");
      return;
    }
    MatrixHandle matrix;
    if (iport->get(matrix) && matrix.get_rep()) {
      matrices.push_back(matrix);
    }
    else {
      //error("Cannot get matrix from input port.");
      //      return;
    }
  }
  
  ColorMapIPort *imap = (ColorMapIPort *)get_iport("ColorMap");
  if (!imap->get(cmap_)) { 
    warning("No input color map.");
    return;
  }
  
  bool do_3d = false;
  if (gui_mode_.get() == string("3D")) do_3d = true;

  gui_color_by_val_.set(1);

  switch (gui_cmode_.get()){
  case 0: color_mode_ = COLOR_BY_VAL; break;
  case 1: color_mode_ = COLOR_BY_ROW; break;
  default:
  case 2: color_mode_ = COLOR_BY_COL; break;
  }
    


  
  bool recompute_geom = false;

  if (cmap_->generation != cgen_) {
    recompute_geom = true;
    cgen_ = cmap_->generation;
  }    
  
  for (unsigned int m = 0; m < matrices.size(); m++) {
    if (recompute_geom) break;
    const MatrixData &data = get_matrix_data(matrices[m]);
    if (data.changed) recompute_geom = true;
  }

  if (cached_gui_cmode_ != gui_cmode_.get()) {
    recompute_geom = true;
    cached_gui_cmode_ = gui_cmode_.get();
  }

  if (cached_gui_gmode_ != gui_gmode_.get()) {
    recompute_geom = true;
    cached_gui_gmode_ = gui_gmode_.get();
  }

  if (cached_gui_showtext_ != gui_showtext_.get()) {
    recompute_geom = true;
    cached_gui_showtext_ = gui_showtext_.get();
  }


  if (recompute_geom) {
    GeomGroup *plot = scinew GeomGroup();
    GeomSwitch *grid = scinew GeomSwitch(scinew GeomMaterial(generate_grid(do_3d), white_));
    plot->add(grid);
    //plot->add(generate_contour(matrices[m]));

    for (unsigned int m = 0; m < matrices.size(); m++) {
      switch (gui_gmode_.get()){
      case 1:
	plot->add(generate_line_graph(matrices[m]));
	break;

      case 2:
	plot->add(generate_3d_bar_graph(matrices[m]));
	break;

      case 3:
	plot->add(generate_3d_sheet_graph(matrices[m]));
	break;

      case 4:
	plot->add(generate_3d_ribbon_graph(matrices[m],false));
	break;
      default:
      case 5:
	plot->add(generate_3d_ribbon_graph(matrices[m],true));
	break;
      }
      if (gui_showtext_.get()) plot->add(generate_text(matrices[m]));
    }
    plot_ = plot;
  }

  //  if (!plot_.get_rep()) return;
  GeomTransform *trans = scinew GeomTransform(plot_);
  double scale = gui_scale_.get();
  trans->scale (Vector(scale, scale, scale));//gui_scale_.get()2.0, 2.0, 2.0));
  trans->translate(Vector(gui_trans_x_.get(), gui_trans_y_.get(), -1.0));
    
  GeomHandle obj = trans;
  if (!do_3d) obj = scinew GeomSticky(trans);
    
  GeometryOPort *ogeom = (GeometryOPort *)get_oport("Geometry");
  ogeom->delAll();
  ogeom->addObj(obj, "ShowMatrix Plot" );
  ogeom->flushViews();

}

const ShowMatrix::MatrixData &
ShowMatrix::get_matrix_data(MatrixHandle mh)
{
  Matrix *m = mh.get_rep();
  map<Matrix *, MatrixData>::iterator iter = matrix_data_.find(m);

  if (iter == matrix_data_.end() ||
      (*iter).second.generation != m->generation ||
      (*iter).second.row_begin != gui_row_begin_.get() ||
      (*iter).second.row_end   != gui_row_end_.get() ||
      (*iter).second.col_begin != gui_col_begin_.get() ||
      (*iter).second.col_end   != gui_col_end_.get()) {

    // Some matrix data value range has changed, recompute them all!
    MatrixData &data = matrix_data_[m];
    data.generation = m->generation;

    data.row_begin  = gui_row_begin_.get();
    data.row_end    = gui_row_end_.get();
    // Bad row indicies
    if (data.row_begin >= m->nrows() ||
	data.row_end < 0 ||
	data.row_begin > data.row_end)
    {
      data.row_begin = 0;
      data.row_end = m->nrows()-1;
    }
    
    // If for some reason we get a start value less than 0, set to 0
    if (data.row_begin < 0) data.row_begin = 0;
    
    // If the end index is beyond the matrix, set to last index
    if (data.row_end >= m->nrows()) data.row_end = mh->nrows()-1;

    gui_row_begin_.set(data.row_begin);
    gui_row_end_.set(data.row_end);


    data.col_begin  = gui_col_begin_.get();
    data.col_end    = gui_col_end_.get();
    // Bad col indicies
    if (data.col_begin >= m->ncols() ||
	data.col_end < 0 ||
	data.col_begin > data.col_end)
    {
      data.col_begin = 0;
      data.col_end = m->ncols()-1;
    }
    
    // If for some reason we get a start value less than 0, set to 0
    if (data.col_begin < 0) data.col_begin = 0;
    
    // If the end index is beyond the matrix, set to last index
    if (data.col_end >= m->ncols()) data.col_end = mh->ncols()-1;

    gui_col_begin_.set(data.col_begin);
    gui_col_end_.set(data.col_end);


    data.min = get_value(mh, data.row_begin, data.col_begin);
    data.max = data.min;
    
    for (int row = data.row_begin; row <= data.row_end; row++)
      for (int col = data.col_begin; col <= data.col_end; col++) {
	const double val = get_value(mh, row, col);
	if (val < data.min) data.min = val;
	if (val > data.max) data.max = val;
      }
    data.changed = true;
    return data;
  }
  else 
  {
    (*iter).second.changed = false;
    return (*iter).second;
  }
}



} // End namespace SCIRun


#if 0

  GeomGrid *grid2 = scinew GeomGrid(10,10,
				    Point(.0,.0,.0),
				    Vector(2.0,0.0,0.0),
				    Vector(0.0,0.0,2.0));
  ogeom->addObj(grid2, "Grid");

    // set params from properties before drawing leads
    std::ostringstream buffer;
    buffer.precision(2);
    grid_text->set_font_index(3);
    PropertyManager *pm = mh.get_rep();
    string units;
    double xstart = 0.0;
    double xend = 1.0;
    if (pm && pm->get_property(string("time-units"), units)) {
      Point p(-0.1, -0.1, -0.1);
      grid_text->add(units, p);
    }  
    if (pm && pm->get_property(string("time-start"), xstart)) {
      
      buffer.str("");
      buffer << xstart;
      Point p(0., 0., 0.);
      grid_text->add(buffer.str(), p);
    }  
    //    if (pm && pm->get_property(string("time-end"), end)) {
    //  tmax_.set(end);
    //}  
    Point p(0., 0., 0.);
    grid_text->add("ORIGIN", p);
#endif


