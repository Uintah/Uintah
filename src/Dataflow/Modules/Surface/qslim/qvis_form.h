#ifndef FD_view_panel_h_
#define FD_view_panel_h_
/* Header file generated with fdesign. */

/**** Callback routines ****/

extern void cb_snapshot(FL_OBJECT *, long);
extern void cb_decimate(FL_OBJECT *, long);
extern void cb_save_model(FL_OBJECT *, long);


/**** Forms and Objects ****/

typedef struct {
	FL_FORM *view_panel;
	FL_OBJECT *view;
	FL_OBJECT *status_line;
	FL_OBJECT *face_target;
	void *vdata;
	long ldata;
} FD_view_panel;

extern FD_view_panel * create_form_view_panel(void);

#endif /* FD_view_panel_h_ */
