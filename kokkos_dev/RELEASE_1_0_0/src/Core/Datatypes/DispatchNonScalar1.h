//  This is a generated file, DO NOT EDIT!!

#if !defined(Datatypes_Dispatch1_h)
#define Datatypes_Dispatch1_h

// dispatch_non_scalar1 macro follows
#define dispatch_non_scalar1(field1, callback)\
  bool disp_error = false;\
  string disp_msg;\
  string disp_name = field1->get_type_name(0);\
  if (disp_name == "TetVol") {\
    if (field1->get_type_name(1) == "Vector") {\
      TetVol<Vector> *f1 = 0;\
      f1 = dynamic_cast<TetVol<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      TetVol<Tensor> *f1 = 0;\
      f1 = dynamic_cast<TetVol<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TetVol<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "LatticeVol") {\
    if (field1->get_type_name(1) == "Vector") {\
      LatticeVol<Vector> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      LatticeVol<Tensor> *f1 = 0;\
      f1 = dynamic_cast<LatticeVol<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LatticeVol<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "ContourField") {\
    if (field1->get_type_name(1) == "Vector") {\
      ContourField<Vector> *f1 = 0;\
      f1 = dynamic_cast<ContourField<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      ContourField<Tensor> *f1 = 0;\
      f1 = dynamic_cast<ContourField<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "ContourField<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "TriSurf") {\
    if (field1->get_type_name(1) == "Vector") {\
      TriSurf<Vector> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      TriSurf<Tensor> *f1 = 0;\
      f1 = dynamic_cast<TriSurf<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "TriSurf<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_name == "PointCloud") {\
    if (field1->get_type_name(1) == "Vector") {\
      PointCloud<Vector> *f1 = 0;\
      f1 = dynamic_cast<PointCloud<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "PointCloud<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Tensor") {\
      PointCloud<Tensor> *f1 = 0;\
      f1 = dynamic_cast<PointCloud<Tensor>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "PointCloud<Tensor>::get_type_name is broken";\
      }\
    }\
  } else if (disp_error) {\
    cerr << "Error: " << disp_msg << endl;\
  }\

#endif //Datatypes_Dispatch1_h
