//  This is a generated file, DO NOT EDIT!!

#if !defined(Datatypes_Dispatch1_h)
#define Datatypes_Dispatch1_h

// uintah_dispatch_scalar1 macro follows
#define uintah_dispatch_scalar1(field1, callback)\
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
    } else if (field1->get_type_name(1) == "int") {\
      LevelField<int> *f1 = 0;\
      f1 = dynamic_cast<LevelField<int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LevelField<int>::get_type_name is broken";\
      }\
    } else if (field1->get_type_name(1) == "long int") {\
      LevelField<long int> *f1 = 0;\
      f1 = dynamic_cast<LevelField<long int>*>(field1.get_rep());\
      if (f1) {\
        callback(f1);\
      } else {\
        disp_error = true; disp_msg = "LevelField<long int>::get_type_name is broken";\
      }\
    }\
  } else if (disp_error) {\
    cerr << "Error: " << disp_msg << endl;\
  }\

#endif //Datatypes_Dispatch1_h
