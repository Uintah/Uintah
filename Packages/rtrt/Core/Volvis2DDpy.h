#ifndef __2DVOLVISDPY_H__
#define __2DVOLVISDPY_H__

#include <Packages/rtrt/Core/DpyBase.h>
#include <Core/Thread/Runnable.h>
#include <Packages/rtrt/Core/shape.h>
#include <Packages/rtrt/Core/texture.h>
#include <Packages/rtrt/Core/widget.h>
#include <vector>
#include <Packages/rtrt/Core/VolumeVis2D.h>

// used only for rendering hack
#define CLEAN 0
#define FAST 1

using std::vector;

namespace rtrt {
  //  template<class T>
    class Volvis2DDpy : public DpyBase {
    // creates the background texture
    virtual void createBGText(float vmin, float vmax, float gmin, float gmax);
    // restores visible background texture to the clean original
    virtual void loadCleanTexture();
    // draws the background texture
    virtual void drawBackground();
    // adds a new widget to the end of the vector
    virtual void addWidget( int x, int y );
    // cycles through widget types: tri->rect(ell)->rect(1d)->rect(deft)->tri..
    virtual void cycleWidgets( int type );
    // draws all widgets in widgets vector without their textures
    virtual void drawWidgets( GLenum mode );
    // paints widget textures onto the background
    virtual void bindWidgetTextures();
    // determines whether a pixel is inside of a widget
    virtual bool insideAnyWidget( int x, int y );
    // moves user-selected widget to end of widgets vector to be drawn last
    virtual void prioritizeWidgets();
    // retrieves information about picked widgets, determines which widget was picked
    virtual void processHits( GLint hits, GLuint buffer[] );
    // determines which widget the user picked
    virtual void pickShape( int x, int y );


    // Called at the start of run.
    virtual void init();
    // Called whenever the window needs to be redrawn
    virtual void display();
    // Called when the window is resized.  Note: xres and yres will not be
    // updated by the event handler.  That's what this function is for.
    virtual void resize( const int width, const int height );
    // Key is pressed/released.  Use the XK_xxx constants to determine
    // which key was pressed/released
    virtual void key_pressed( unsigned long key );
    // These handle mouse button events.  button indicates which button.
    // x and y are the location measured from the upper left corner of the
    // window.
    virtual void button_pressed(MouseButton button, const int x, const int y);
    virtual void button_released(MouseButton button, const int x, const int y);
    virtual void button_motion(MouseButton button, const int x, const int y);
    
  
  public:
    struct voxel_valuepair {
      int value;
      int gradient;
    };
    // maintains cutplane voxels being examined
    vector<voxel_valuepair*> cp_voxels;

    // Deletes cutplane voxel information
    void delete_voxel_storage();
    // Called whenever the histogram borders need to be redrawn
    void display_hist_perimeter();
    // Called whenever the sliders need to be redrawn
    void display_controls();
    // Called whenever the user adjusts the master opacity control
    void adjustMasterOpacity( float x );
    // Called whenever the user adjusts the cutplane opacity
    void adjustCutplaneOpacity( float x );
    // Called whenever the user adjusts the cutplane grayscale
    void adjustCutplaneGS( float x );
    // Stores a voxel's gradient/value pair
    void store_voxel( Voxel2D<float> voxel );
    // Displays cutplane voxels on the histogram
    void display_cp_voxels();
    // Assigns a color and opacity corresponding to a given voxel
    void voxel_lookup( Voxel2D<float> voxel, Color &color, float &opacity );
    // Attaches a volume to be rendered through this user interface
    void attach( VolumeVis2D *volume );
    // Loads a previously saved user interface setup
    void loadUIState( unsigned long key );
    // Saves a user interface setup
    void saveUIState( unsigned long key );
    // Called whenever the user adjusts the size of the ray increment
    void adjustRaySize( unsigned long key );
    // Used only for a rendering hack
    bool skip_opacity( Voxel2D<float> v1, Voxel2D<float> v2,
		       Voxel2D<float> v3, Voxel2D<float> v4 );
    // Called whenever the user interface is changed
    virtual void animate(bool &changed);


    // if true, only last widget texture must be repainted
    bool widgetsMaintained;
    Texture <GLfloat> *transTexture3; // stores unchanging widget textures

    // maintains user interface parameters to manage menus/graphs/etc
    struct WindowParams {
      float border;
      float height;
      float width;
      float menu_height;
    };
    WindowParams* UIwind;

    // used only for rendering hack
    unsigned int render_mode;

    // keep track of ray size to accurately assign color/opacity values
    float original_t_inc;
    float t_inc;
    float t_inc_diff;

    // whether or not a cutting plane is being used
    bool cut;

    // whether or not any background textures need to be drawn
    bool hist_changed;
    bool transFunc_changed;

    // for adjusting the master opacity control
    bool m_opacity_adjusting;
    float master_opacity;
    GLBar* m_opacity_slider;
    GLBar* m_opacity_bar;

    // for adjusting the cutplane opacity
    bool cp_opacity_adjusting;
    float cp_opacity;
    GLBar* cp_opacity_slider;
    GLBar* cp_opacity_bar;

    // for adjusting the cutplane grayscale
    bool cp_gs_adjusting;
    float cp_gs;
    GLBar* cp_gs_slider;
    GLBar* cp_gs_bar;

    // precomputed values to speed up voxel-to-texture calculation
    float text_x_convert;
    float text_y_convert;

    // the collection of widgets that control the transfer function
    vector<Widget*> widgets;

    int pickedIndex;                   // index of currently selected widget
/*      int old_x;                         // saved most recent x-coordinate */
/*      int old_y;                         // saved most recent y-coordinate */
    float pixel_width;                 // screenspace-to-worldspace x-dim ratio
    float pixel_height;                // screenspace-to-worldspace y-dim ratio

    // voxel minimum/maximum values 
    float vmin, vmax;
    float gmin, gmax;

    // for user-defined histogram parameters
    bool hist_adjust;
    float current_vmin, current_vmax, selected_vmin, selected_vmax;
    float current_gmin, current_gmax, selected_gmin, selected_gmax;

    GLuint bgTextName;                 // histogram texture
    GLuint transFuncTextName;          // transfer function texture
    Texture <GLfloat> *bgTextImage;    // clean background texture
    Texture <GLfloat> *transTexture1;  // visible transfer functions
    Texture <GLfloat> *transTexture2;  // swapped to remove rendering "streaks"

    // collection of volumes being rendered
    vector<VolumeVis2D*> volumes;

    Volvis2DDpy( float t_inc, bool cut );
  };

} // end namespace rtrt

#endif
