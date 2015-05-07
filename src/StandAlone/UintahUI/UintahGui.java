//**************************************************************************
// Program : UintahGui.java
// Purpose : Create a user interface Uintah (for foam creation)
// Author  : Biswajit Banerjee
// Date    : May 3, 2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;

//**************************************************************************
// Class   : UintahGui
// Purpose : Controller routine
//**************************************************************************
public class UintahGui extends JApplet {

  // Data
  private ParticleList d_partList = null;

  private UintahInputPanel uintahInputPanel = null;
  private ParticleGeneratePanel particleGenPanel = null;
  private JTabbedPane mainTabbedPane = null;

  public HelpAboutFrame helpAboutFrame = null;
  public File oldFile = null;
  public static JFrame mainFrame = null;

  public static int OPEN = 1;
  public static int SAVE = 2;

  // If the applet is called as an application
  public static void main(String[] args) {
    
    // Create the frame
    mainFrame = new JFrame("Uintah User Interface");

    // Add a window listener
    mainFrame.addWindowListener(new WindowAdapter() {
      public void windowClosing(WindowEvent e) {System.exit(0);}
    });

    // instantiate
    UintahGui uintahGui = new UintahGui();
    uintahGui.init();

    // Add the stuff to the frame
    mainFrame.setLocation(20,50);
    mainFrame.setContentPane(uintahGui);
    mainFrame.pack();
    mainFrame.setVisible(true);
  }

  // Constructor
  public UintahGui() {
    this(true);
  }
  public UintahGui(boolean inAnApplet) {
    if (inAnApplet) {
      getRootPane().putClientProperty("defeatSystemEventQueueCheck", 
                                    Boolean.TRUE);
      getRootPane().setLocation(20,50);
    }

    // Create a new Particle list
    d_partList = new ParticleList();
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
    menuItem = new JMenuItem("Read Particle Location Data");
    fileMenu.add(menuItem);
    menuItem.addActionListener(menuListener);

    menuItem = new JMenuItem("Save Uintah Input File");
    fileMenu.add(menuItem);
    menuItem.addActionListener(menuListener);

    menuItem = new JMenuItem("Exit");
    fileMenu.add(menuItem);
    menuItem.addActionListener(menuListener);

    // Create the main tabbed pane
    mainTabbedPane = new JTabbedPane();

    // Create the panels to be added to the tabbed pane
    uintahInputPanel = new UintahInputPanel(d_partList, this);
    particleGenPanel = new ParticleGeneratePanel(d_partList, this);

    // Add the tabs
    mainTabbedPane.addTab("Uintah Inputs", null,
                          uintahInputPanel, null);
    mainTabbedPane.addTab("Generate Particle Locations", null,
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

    // Create the Tab Listener
    TabListener tabListener = new TabListener();
    mainTabbedPane.addChangeListener(tabListener);

  }

  // Update panels
  public void updatePanels() {
    mainFrame.pack();
  }

  // Menu listener
  private class MenuListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      JMenuItem source = (JMenuItem)(e.getSource());
      String text = source.getText();
      if (text.equals("Exit")) {
        System.exit(0);
      } else if (text.equals("Read Particle Location Data")) {
        File particleFile = null;
        if ((particleFile = getFileName(UintahGui.OPEN)) != null) {
          d_partList.readFromFile(particleFile);
        }
      } else if (text.equals("Save Uintah Input File")) {
        File uintahFile = null;
        if ((uintahFile = getFileName(UintahGui.SAVE)) != null) {
          writeUintah(uintahFile);
        }
      } else if (text.equals("About")) {
        helpAboutFrame.setVisible(true);
      }
    }
  }

  // Tab listener
  private class TabListener implements ChangeListener {
    public void stateChanged(ChangeEvent e) {
      int curTab = mainTabbedPane.getSelectedIndex();
      if (curTab == 0) {
        particleGenPanel.setVisibleDisplayFrame(false);
      } else {
        particleGenPanel.setVisibleDisplayFrame(true);
        uintahInputPanel.setVisibleDisplayFrame(false);
      }
    }
  }

  // Get the name of the file 
  private File getFileName(int option) {
    JFileChooser fc = new JFileChooser(new File(".."));
    if (oldFile != null) fc.setSelectedFile(oldFile);
    int returnVal = 0; 
    if (option == UintahGui.OPEN) {
      returnVal = fc.showOpenDialog(UintahGui.this);
    } else {
      returnVal = fc.showSaveDialog(UintahGui.this);
    }
    if (returnVal == JFileChooser.APPROVE_OPTION) {
      File file = fc.getSelectedFile();
      oldFile = file;
      return file;
    } else return null;
  }

  // Write the output in Uintah format
  private void writeUintah(File outputFile) {
    
    // Create filewriter and printwriter
    try {
      FileWriter fw = new FileWriter(outputFile);
      PrintWriter pw = new PrintWriter(fw);

      uintahInputPanel.writeUintah(pw);

      pw.close();
      fw.close();

    } catch (Exception event) {
      System.out.println("Could not write to file "+outputFile.getName());
    }
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

  // For setting the gridbagconstraints for this application
  public static void setConstraints(GridBagConstraints c, int col, int row) {
    c.fill = GridBagConstraints.NONE;
    c.weightx = 1.0;
    c.weighty = 1.0;
    c.gridx = col;
    c.gridy = row;
    c.gridwidth = 1;
    c.gridheight = 1;
    Insets insets = new Insets(5, 5, 5, 5);
    c.insets = insets;
  }

  // For setting the gridbagconstraints for this application
  public static void setConstraints(GridBagConstraints c, int fill,
                                    int col, int row) {
    c.fill = fill;
    c.weightx = 1.0;
    c.weighty = 1.0;
    c.gridx = col;
    c.gridy = row;
    c.gridwidth = 1;
    c.gridheight = 1;
    Insets insets = new Insets(5, 5, 5, 5);
    c.insets = insets;
  }

  // For setting the gridbagconstraints for this application
  public static void setConstraints(GridBagConstraints c, int fill,
                                    int xinset, int yinset,
                                    int col, int row) {
    c.fill = fill;
    c.weightx = 1.0;
    c.weighty = 1.0;
    c.gridx = col;
    c.gridy = row;
    c.gridwidth = 1;
    c.gridheight = 1;
    Insets insets = new Insets(yinset, xinset, yinset, xinset);
    c.insets = insets;
  }


}
