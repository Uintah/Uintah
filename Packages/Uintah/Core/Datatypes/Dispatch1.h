//  This is a generated file, DO NOT EDIT!!

#if !defined(Datatypes_Dispatch1_h)
#define Datatypes_Dispatch1_h

// dispatch1 macro follows
#define uintah_dispatch1(field1, callback)\
  bool disp_error = false;\
  string disp_msg;\
  string disp_name = field1->get_type_name(0);\
  if (disp_name == "LevelField") {\
    if (field1->get_type_name(1) == "double") {\
      LevelField<double> *f1 = 0;\
      f1 = dynamic_cast<LevelField<double>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LevelField<double>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "long") {\
      LevelField<long> *f1 = 0;\
      f1 = dynamic_cast<LevelField<long>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LevelField<long>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "float") {\
      LevelField<float> *f1 = 0;\
      f1 = dynamic_cast<LevelField<float>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LevelField<float>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Vector") {\
      LevelField<Vector> *f1 = 0;\
      f1 = dynamic_cast<LevelField<Vector>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LevelField<Vector>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "Matrix3") {\
      LevelField<Matrix3> *f1 = 0;\
      f1 = dynamic_cast<LevelField<Matrix3>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LevelField<Matrix3>::get_type_name is broken";\
      }\
    }\
  } else if (disp_error) {\
    cerr << "Error: " << disp_msg << endl;\
  }\

#endif //Datatypes_Dispatch1_h
