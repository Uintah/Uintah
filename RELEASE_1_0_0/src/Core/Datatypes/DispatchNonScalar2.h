//  This is a generated file, DO NOT EDIT!!

#if !defined(Datatypes_Dispatch2_h)
#define Datatypes_Dispatch2_h

// dispatch_non_scalar2 macro follows
#define dispatch_non_scalar2(field1, field2, callback)\
  bool disp_error = false;\
  string disp_msg;\
  string disp_name = field2->get_type_name(0);\
  if (disp_name == "TetVol") {\
    if (field2->get_type_name(1) == "Vector") {\
      TetVol<Vector> *f2 = 0;\
      f2 = dynamic_cast<TetVol<Vector>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
      }\
    } else if (field2->get_type_name(1) == "Tensor") {\
      TetVol<Tensor> *f2 = 0;\
      f2 = dynamic_cast<TetVol<Tensor>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "LatticeVol") {\
    if (field2->get_type_name(1) == "Vector") {\
      LatticeVol<Vector> *f2 = 0;\
      f2 = dynamic_cast<LatticeVol<Vector>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
      }\
    } else if (field2->get_type_name(1) == "Tensor") {\
      LatticeVol<Tensor> *f2 = 0;\
      f2 = dynamic_cast<LatticeVol<Tensor>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "ContourField") {\
    if (field2->get_type_name(1) == "Vector") {\
      ContourField<Vector> *f2 = 0;\
      f2 = dynamic_cast<ContourField<Vector>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
      }\
    } else if (field2->get_type_name(1) == "Tensor") {\
      ContourField<Tensor> *f2 = 0;\
      f2 = dynamic_cast<ContourField<Tensor>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "TriSurf") {\
    if (field2->get_type_name(1) == "Vector") {\
      TriSurf<Vector> *f2 = 0;\
      f2 = dynamic_cast<TriSurf<Vector>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
      }\
    } else if (field2->get_type_name(1) == "Tensor") {\
      TriSurf<Tensor> *f2 = 0;\
      f2 = dynamic_cast<TriSurf<Tensor>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "PointCloud") {\
    if (field2->get_type_name(1) == "Vector") {\
      PointCloud<Vector> *f2 = 0;\
      f2 = dynamic_cast<PointCloud<Vector>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
      }\
    } else if (field2->get_type_name(1) == "Tensor") {\
      PointCloud<Tensor> *f2 = 0;\
      f2 = dynamic_cast<PointCloud<Tensor>*>(field2.get_rep());\
      if (f2) {\
        if (disp_name == "TetVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            TetVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TetVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "LatticeVol") {\
          if (field1->get_type_name(1) == "Vector") {\
            LatticeVol<Vector> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            LatticeVol<Tensor> *f1 = 0;\
            f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "ContourField") {\
          if (field1->get_type_name(1) == "Vector") {\
            ContourField<Vector> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            ContourField<Tensor> *f1 = 0;\
            f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "TriSurf") {\
          if (field1->get_type_name(1) == "Vector") {\
            TriSurf<Vector> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            TriSurf<Tensor> *f1 = 0;\
            f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_name == "PointCloud") {\
          if (field1->get_type_name(1) == "Vector") {\
            PointCloud<Vector> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
            }\
          } else if (field1->get_type_name(1) == "Tensor") {\
            PointCloud<Tensor> *f1 = 0;\
            f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
            if (f1) {\
              callback(f1, f2);\
            } else {\
              disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
            }\
          }\
        } else if (disp_error) {\
          cerr << "Error: " << disp_msg << endl;\
        }\
      } else {\
        disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_error) {\
    cerr << "Error: " << disp_msg << endl;\
  }\

#endif //Datatypes_Dispatch2_h
