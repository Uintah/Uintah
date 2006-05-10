//**************************************************************************
// Program : UintahGui.java
// Purpose : Create a user interface Uintah (for foam creation)
// Author  : Biswajit Banerjee
// Date    : May 3, 2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import javax.swing.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.filechooser.*;

//**************************************************************************
// Class   : UintahGui
// Purpose : Controller routine
//**************************************************************************
public class UintahGui extends JApplet {

  // Data
  private boolean inAnApplet = true;
  public ParticleSize partSizeDist = null;
  public ParticleList partList = null;
  public HelpAboutFrame helpAboutFrame = null;
  public File oldFile = null;

  // Constructor
  public UintahGui() {
    this(true);
  }
  public UintahGui(boolean inAnApplet) {
    this.inAnApplet = inAnApplet;
    if (inAnApplet) {
      getRootPane().putClientProperty("defeatSystemEventQueueCheck", 
				    Boolean.TRUE);
      getRootPane().setLocation(20,50);
    }

    // Create a new Particle list
    partList = new ParticleList();

    // Create a new ParticleSize object
    partSizeDist = new ParticleSize();

  }

  // The init method
  public void init() {

    // Set the look and feel
    try {
      UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
    } catch (Exception e) {
      return;
    }
    
    // Create the menuListener
    MenuListener menuListener = new MenuListener();

    // Create the menu bar
    JMenuBar menuBar = new JMenuBar();
    setJMenuBar(menuBar);

    // Create the file menu
    JMenu fileMenu = new JMenu("File");
    menuBar.add(fileMenu);

    // Create the menuitems
    JMenuItem menuItem;
    menuItem = new JMenuItem("Read Bubble Size Data");
    fileMenu.add(menuItem);
    menuItem.addActionListener(menuListener);

    menuItem = new JMenuItem("Save Uintah Input File");
    fileMenu.add(menuItem);
    menuItem.addActionListener(menuListener);

    menuItem = new JMenuItem("Exit");
    fileMenu.add(menuItem);
    menuItem.addActionListener(menuListener);

    // Create the main tabbed pane
    JTabbedPane mainTabbedPane = new JTabbedPane();

    // Create the panels to be added to the tabbed pane
    UintahInputPanel uintahInputPanel = new UintahInputPanel(partList);
    ParticleGeneratePanel particleGenPanel = new ParticleGeneratePanel(partList);

    // Add the tabs
    mainTabbedPane.addTab("Uintah Inputs", null,
                          uintahInputPanel, null);
    mainTabbedPane.addTab("Generate Bubble Locations", null,
                          particleGenPanel, null);
    mainTabbedPane.setSelectedIndex(0);
    getContentPane().add(mainTabbedPane);

    // Create the help menu
    JMenu helpMenu = new JMenu("Help");
    menuBar.add(helpMenu);

    // Create the menuitems
    menuItem = new JMenuItem("About");
    helpMenu.add(menuItem);
    menuItem.addActionListener(menuListener);

    // Create the invisible help frames
    helpAboutFrame = new HelpAboutFrame();
    helpAboutFrame.pack();
  }

  // If the applet is called as an application
  public static void main(String[] args) {
    
    // Create the frame
    JFrame frame = new JFrame("Uintah User Interface");

    // Add a window listener
    frame.addWindowListener(new WindowAdapter() {
      public void windowClosing(WindowEvent e) {System.exit(0);}
    });

    // instantiate
    UintahGui uintahGui = new UintahGui();
    uintahGui.init();

    // Add the stuff to the frame
    frame.setLocation(20,50);
    frame.setContentPane(uintahGui);
    frame.pack();
    frame.setVisible(true);
  }

  // For setting the gridbagconstraints for this application
  public static void setConstraints(GridBagConstraints c, int fill, double wx,
                double wy, int gx, int gy, int gw, int gh, int ins) {
    c.fill = fill;
    c.weightx = (float) wx;
    c.weighty = (float) wy;
    c.gridx = gx;
    c.gridy = gy;
    c.gridwidth = gw;
    c.gridheight = gh;
    Insets insets = new Insets(ins, ins, ins, ins);
    c.insets = insets;
  }

  class MenuListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      JMenuItem source = (JMenuItem)(e.getSource());
      String text = source.getText();
      if (text.equals("Exit")) {
	System.exit(0);
      } else if (text.equals("Read Bubble Size Data")) {
	File particleFile = null;
	if ((particleFile = getParticleFile()) != null) {
	  //System.out.println("File = "+particleFile.getName());
	  partList.readFromFile(particleFile);
	}
      } else if (text.equals("Save Uintah Input File")) {
	File particleFile = null;
	if ((particleFile = getParticleFile()) != null) {
	  partList.readFromFile(particleFile, 1);
	  partList.writeUintah(particleFile);
	}
      } else if (text.equals("About")) {
	helpAboutFrame.setVisible(true);
      }
    }
  }

  // Get the name of the file containing the particle co-ordinates
  private File getParticleFile() {
    JFileChooser fc = new JFileChooser(new File(".."));
    if (oldFile != null) fc.setSelectedFile(oldFile);
    int returnVal = fc.showOpenDialog(UintahGui.this);
    if (returnVal == JFileChooser.APPROVE_OPTION) {
      File file = fc.getSelectedFile();
      oldFile = file;
      return file;
    } else return null;
  }

}
