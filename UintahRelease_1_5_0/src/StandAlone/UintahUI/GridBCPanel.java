//**************************************************************************
// Program : GridBCPanel.java
// Purpose : Take inputs for the grid and boundary conditions
// Author  : Biswajit Banerjee
// Date    : 05/28/2006
// Mods    :
//**************************************************************************

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.PrintWriter;

public class GridBCPanel extends JPanel {

  // Data
  private double d_domainSize;
  private double d_numLevel;

  private UintahInputPanel d_parentPanel = null;

  private JTabbedPane levelTabbedPane = null;
  private JTabbedPane bcTabbedPane = null;
  private LevelPanel levelPanel = null;
  private BCPanel xminPanel = null;
  private BCPanel xmaxPanel = null;
  private BCPanel yminPanel = null;
  private BCPanel ymaxPanel = null;
  private BCPanel zminPanel = null;
  private BCPanel zmaxPanel = null;

  // Constructor
  public GridBCPanel(UintahInputPanel parentPanel) {

    // Copy the arguments
    d_parentPanel = parentPanel;
    d_domainSize = 100.0;
    d_numLevel = 1;

    // Create a grid bag
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);
    int fillBoth = GridBagConstraints.BOTH;
    int fill = GridBagConstraints.NONE;
    int xgap = 5;
    int ygap = 0;

    // Create a tabbed pane for the levels
    levelTabbedPane = new JTabbedPane();

    // Create and add level to tabbed pane
    int level = 0;
    String levelID = new String("Level "+String.valueOf(level));
    levelPanel = new LevelPanel(level);
    levelTabbedPane.addTab(levelID, null, levelPanel, null);
    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 0);
    gb.setConstraints(levelTabbedPane, gbc);
    add(levelTabbedPane);

    // Boundary conditions
    bcTabbedPane = new JTabbedPane();

    // Create and add level to tabbed pane
    xminPanel = new BCPanel("x-");
    bcTabbedPane.addTab("BCs at x-", null, xminPanel, null);

    xmaxPanel = new BCPanel("x+");
    bcTabbedPane.addTab("BCs at x+", null, xmaxPanel, null);

    yminPanel = new BCPanel("y-");
    bcTabbedPane.addTab("BCs at y-", null, yminPanel, null);

    ymaxPanel = new BCPanel("y+");
    bcTabbedPane.addTab("BCs at y+", null, ymaxPanel, null);

    zminPanel = new BCPanel("z-");
    bcTabbedPane.addTab("BCs at z-", null, zminPanel, null);

    zmaxPanel = new BCPanel("z+");
    bcTabbedPane.addTab("BCs at z+", null, zmaxPanel, null);

    UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 2);
    gb.setConstraints(bcTabbedPane, gbc);
    add(bcTabbedPane);
  }

  public void refresh() {
  }

  public void setDomainSize(double domainSize) {
    d_domainSize = domainSize;
  }

  public double getDomainSize() {
    return d_domainSize;
  }

  public void writeUintah(PrintWriter pw, String tab) {

    String tab1 = new String(tab+"  ");
    pw.println(tab+"<Level>");
    levelPanel.writeUintah(pw, tab1);
    pw.println(tab+"</Level>");
    pw.println(tab+"<BoundaryConditions>");
    xminPanel.writeUintah(pw, tab1);
    xmaxPanel.writeUintah(pw, tab1);
    yminPanel.writeUintah(pw, tab1);
    ymaxPanel.writeUintah(pw, tab1);
    zminPanel.writeUintah(pw, tab1);
    zmaxPanel.writeUintah(pw, tab1);
    pw.println(tab+"</BoundaryConditions>");
  }

  private class LevelPanel extends JPanel {

    // Local data
    int d_level = 0;

    // Local components
    private DecimalVectorField minEntry = null;
    private DecimalVectorField maxEntry = null;
    private IntegerVectorField resEntry = null;
    private IntegerVectorField extraEntry = null;
    private IntegerVectorField patchEntry = null;

    public LevelPanel(int level) {

      // Save data
      d_level = level;

      // Create a grid bag
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);
      int fillBoth = GridBagConstraints.BOTH;
      int fill = GridBagConstraints.NONE;
      int xgap = 5;
      int ygap = 0;

      // Input domain limits
      JLabel minLabel = new JLabel("Domain Lower Left Corner");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 0);
      gb.setConstraints(minLabel, gbc);
      add(minLabel);

      minEntry = new DecimalVectorField(0.0, 0.0, 0.0, 5, true);
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 0);
      gb.setConstraints(minEntry, gbc);
      add(minEntry);

      JLabel maxLabel = new JLabel("Domain Upper Right Corner");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 1);
      gb.setConstraints(maxLabel, gbc);
      add(maxLabel);

      maxEntry = new DecimalVectorField(0.0, 0.0, 0.0, 5, true);
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 1);
      gb.setConstraints(maxEntry, gbc);
      add(maxEntry);

      // Input resolution
      JLabel resLabel = new JLabel("Grid Resolution");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 2);
      gb.setConstraints(resLabel, gbc);
      add(resLabel);

      resEntry = new IntegerVectorField(0, 0, 0, 5);
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 2);
      gb.setConstraints(resEntry, gbc);
      add(resEntry);

      // Input extra cells
      JLabel extraLabel = new JLabel("Extra Grid Cells");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 3);
      gb.setConstraints(extraLabel, gbc);
      add(extraLabel);

      extraEntry = new IntegerVectorField(0, 0, 0, 5);
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 3);
      gb.setConstraints(extraEntry, gbc);
      add(extraEntry);

      // Input patches
      JLabel patchLabel = new JLabel("Patches");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 4);
      gb.setConstraints(patchLabel, gbc);
      add(patchLabel);

      patchEntry = new IntegerVectorField(1, 1, 1, 5);
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 4);
      gb.setConstraints(patchEntry, gbc);
      add(patchEntry);

    }

    public void writeUintah(PrintWriter pw, String tab) {
      String tab1 = new String(tab+"  ");
      pw.println(tab+"<Box label=\"Level"+d_level+"\">");
      pw.println(tab1+"<lower> ["+minEntry.x()+", "+minEntry.y()+", "+
                 minEntry.z()+ "] </lower>");
      pw.println(tab1+"<upper> ["+maxEntry.x()+", "+maxEntry.y()+", "+
                 maxEntry.z()+ "] </upper>");
      pw.println(tab1+"<resolution> ["+resEntry.x()+", "+resEntry.y()+", "+
                 resEntry.z()+ "] </resolution>");
      pw.println(tab1+"<extraCells> ["+extraEntry.x()+", "+extraEntry.y()+", "+
                 extraEntry.z()+"] </extraCells>");
      pw.println(tab1+"<patches> ["+patchEntry.x()+", "+patchEntry.y()+", "+
                 patchEntry.z()+"] </patches>");
      pw.println(tab+"</Box>");
    }
  }

  private class BCPanel extends JPanel {

    // Local data
    private String d_location = null;
    private boolean d_symm = false;

    private SymmBCPanel symmBCPanel = null;
    private ScalarBCPanel pressurePanel = null;
    private ScalarBCPanel densityPanel = null;
    private ScalarBCPanel temperaturePanel = null;
    private ScalarBCPanel spVolPanel = null;
    private VectorBCPanel velocityPanel = null;

    public BCPanel(String location) {

      // Initialize
      d_symm = false;
      d_location = location;

      // Create a grid bag
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);
      int fillBoth = GridBagConstraints.BOTH;
      int fill = GridBagConstraints.NONE;
      int xgap = 5;
      int ygap = 0;

      symmBCPanel = new SymmBCPanel(location);
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 0);
      gb.setConstraints(symmBCPanel, gbc);
      add(symmBCPanel);

      pressurePanel = new ScalarBCPanel("Pressure");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 1);
      gb.setConstraints(pressurePanel, gbc);
      add(pressurePanel);

      densityPanel = new ScalarBCPanel("Density");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 2);
      gb.setConstraints(densityPanel, gbc);
      add(densityPanel);

      temperaturePanel = new ScalarBCPanel("Temperature");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 3);
      gb.setConstraints(temperaturePanel, gbc);
      add(temperaturePanel);

      spVolPanel = new ScalarBCPanel("SpecificVol");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 4);
      gb.setConstraints(spVolPanel, gbc);
      add(spVolPanel);

      velocityPanel = new VectorBCPanel("Velocity");
      UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 5);
      gb.setConstraints(velocityPanel, gbc);
      add(velocityPanel);

    }
   
    public void writeUintah(PrintWriter pw, String tab) {
      String tab1 = new String(tab+"  ");
      pw.println(tab+"<Face side=\""+d_location+"\">");
      if (d_symm) {
        symmBCPanel.writeUintah(pw, tab1);
      } else {
        pressurePanel.writeUintah(pw, tab1);
        densityPanel.writeUintah(pw, tab1);
        temperaturePanel.writeUintah(pw, tab1);
        spVolPanel.writeUintah(pw, tab1);
        velocityPanel.writeUintah(pw, tab1);
      }
      pw.println(tab+"</Face>");
    }

    private class SymmBCPanel extends JPanel 
                              implements ItemListener {

      // Local data
      private JCheckBox symmCB = null;

      public SymmBCPanel(String location) {

        // Create a grid bag
        GridBagLayout gb = new GridBagLayout();
        GridBagConstraints gbc = new GridBagConstraints();
        setLayout(gb);
        int fillBoth = GridBagConstraints.BOTH;
        //int fill = GridBagConstraints.NONE;
        int xgap = 5;
        int ygap = 0;

        // Label
        symmCB = new JCheckBox("Symmetry BCs only");
        symmCB.setSelected(false);
        symmCB.addItemListener(this);

        UintahGui.setConstraints(gbc, fillBoth, xgap, ygap, 0, 0);
        gb.setConstraints(symmCB, gbc);
        add(symmCB);
      }

      public void itemStateChanged(ItemEvent e) {
        if (e.getStateChange() == ItemEvent.SELECTED) {
          d_symm = true;
          pressurePanel.setEnabled(false);
          densityPanel.setEnabled(false);
          temperaturePanel.setEnabled(false);
          spVolPanel.setEnabled(false);
          velocityPanel.setEnabled(false);
        } else {
          d_symm = false;
          pressurePanel.setEnabled(true);
          densityPanel.setEnabled(true);
          temperaturePanel.setEnabled(true);
          spVolPanel.setEnabled(true);
          velocityPanel.setEnabled(true);
        }
      }

      public void writeUintah(PrintWriter pw, String tab) {

        String tab1 = new String(tab+"  ");
        pw.println(tab+
            "<BCType id=\"all\" var=\"symmetry\" label=\"Symmetric\">");
        pw.println(tab+"</BCType>");
      }

    }

    private class ScalarBCPanel extends JPanel 
                                implements ActionListener{

      // Local data
      private String d_scalar = null;
      private String d_type = null;
      private String d_mat = null;

      private JLabel presLabel = null;
      private JLabel typeLabel = null;
      private JComboBox typeCB = null;
      private JLabel matLabel = null;
      private JComboBox matCB = null;
      private JLabel valLabel = null;
      private DecimalField valEntry = null;

      public ScalarBCPanel(String scalar) {

        // Initialize
        d_scalar = scalar;
        d_type = "Symmetric";
        d_mat = "all";

        // Create a grid layout
        setLayout(new FlowLayout(FlowLayout.LEFT));

        // Label
        presLabel = new JLabel(scalar+":");
        add(presLabel);

        typeLabel = new JLabel("BC Type");
        add(typeLabel);

        typeCB = new JComboBox();
        typeCB.addItem("Symmetry");
        typeCB.addItem("Dirichlet");
        typeCB.addItem("Neumann");
        typeCB.addItem("Compute From Density");
        typeCB.addActionListener(this);
        add(typeCB);

        matLabel = new JLabel("Material");
        add(matLabel);

        matCB = new JComboBox();
        matCB.addItem("All");
        matCB.addItem("Material 0");
        matCB.addActionListener(this);
        add(matCB);

        valLabel = new JLabel("Value");
        add(valLabel);

        valEntry = new DecimalField(0.0, 9);
        add(valEntry);
      }

      public void actionPerformed(ActionEvent e) {

        JComboBox source = (JComboBox) e.getSource();
        String item = (String) source.getSelectedItem();

        if (source.equals(typeCB)) {
          if (item.equals("Symmetry")) {
            d_type = "Symmetric";
          } else if (item.equals("Dirchlet")) {
            d_type = "Dirichlet";
          } else if (item.equals("Neumann")) {
            d_type = "Neumann";
          } else if (item.equals("Compute From Density")) {
            d_type = "computeFromDensity";
          }
 
        } else if (source.equals(matCB)) {
          if (item.equals("All")) {
            d_mat = "all";
          } else if (item.equals("Material 0")) {
            d_mat = "0";
          }
        }

      }

      public void setEnabled(boolean enable) {
        if (enable) {
          presLabel.setEnabled(true);
          typeLabel.setEnabled(true);
          typeCB.setEnabled(true);
          matLabel.setEnabled(true);
          matCB.setEnabled(true);
          valLabel.setEnabled(true);
          valEntry.setEnabled(true);
        } else {
          presLabel.setEnabled(false);
          typeLabel.setEnabled(false);
          typeCB.setEnabled(false);
          matLabel.setEnabled(false);
          matCB.setEnabled(false);
          valLabel.setEnabled(false);
          valEntry.setEnabled(false);
        }
      }

      public void writeUintah(PrintWriter pw, String tab) {

        String tab1 = new String(tab+"  ");
        pw.println(tab+"<BCType id=\""+d_mat+"\" var=\""+d_type+"\" label=\""+
                     d_scalar+ "\">");
        pw.println(tab1+"<value> "+valEntry.getValue()+" </value>");
        pw.println(tab+"</BCType>");
      }

    }

    private class VectorBCPanel extends JPanel
                                implements ActionListener {

      // Local data
      private String d_vector = null;
      private String d_type = null;
      private String d_mat = null;

      private JLabel presLabel = null;
      private JLabel typeLabel = null;
      private JComboBox typeCB = null;
      private JLabel matLabel = null;
      private JComboBox matCB = null;
      private JLabel valLabel = null;
      private DecimalVectorField valEntry = null;

      public VectorBCPanel(String vector) {

        // Initialize
        d_vector = vector;
        d_type = "Symmetric";
        d_mat = "all";

        // Create a grid bag
        GridBagLayout gb = new GridBagLayout();
        GridBagConstraints gbc = new GridBagConstraints();
        setLayout(gb);
        int fillBoth = GridBagConstraints.BOTH;
        int fill = GridBagConstraints.NONE;
        int xgap = 5;
        int ygap = 0;

        // Label
        presLabel = new JLabel(vector+":");
        UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 0);
        gb.setConstraints(presLabel, gbc);
        add(presLabel);

        typeLabel = new JLabel("BC Type");
        UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 0);
        gb.setConstraints(typeLabel, gbc);
        add(typeLabel);

        typeCB = new JComboBox();
        typeCB.addItem("Symmetry");
        typeCB.addItem("Dirichlet");
        typeCB.addItem("Neumann");
        typeCB.addActionListener(this);
        UintahGui.setConstraints(gbc, fill, xgap, ygap, 2, 0);
        gb.setConstraints(typeCB, gbc);
        add(typeCB);

        matLabel = new JLabel("Material");
        UintahGui.setConstraints(gbc, fill, xgap, ygap, 3, 0);
        gb.setConstraints(matLabel, gbc);
        add(matLabel);

        matCB = new JComboBox();
        matCB.addItem("All");
        matCB.addItem("Material 0");
        matCB.addActionListener(this);
        UintahGui.setConstraints(gbc, fill, xgap, ygap, 4, 0);
        gb.setConstraints(matCB, gbc);
        add(matCB);

        valLabel = new JLabel("Value");
        UintahGui.setConstraints(gbc, fill, xgap, ygap, 5, 0);
        gb.setConstraints(valLabel, gbc);
        add(valLabel);

        valEntry = new DecimalVectorField(0.0, 0.0, 0.0, 9, true);
        UintahGui.setConstraints(gbc, fill, xgap, ygap, 6, 0);
        gb.setConstraints(valEntry, gbc);
        add(valEntry);
      }

      public void actionPerformed(ActionEvent e) {

        JComboBox source = (JComboBox) e.getSource();
        String item = (String) source.getSelectedItem();

        if (source.equals(typeCB)) {
          if (item.equals("Symmetry")) {
            d_type = "Symmetric";
          } else if (item.equals("Dirchlet")) {
            d_type = "Dirichlet";
          } else if (item.equals("Neumann")) {
            d_type = "Neumann";
          }
 
        } else if (source.equals(matCB)) {
          if (item.equals("All")) {
            d_mat = "all";
          } else if (item.equals("Material 0")) {
            d_mat = "0";
          }
        }

      }

      public void setEnabled(boolean enable) {
        if (enable) {
          presLabel.setEnabled(true);
          typeLabel.setEnabled(true);
          typeCB.setEnabled(true);
          matLabel.setEnabled(true);
          matCB.setEnabled(true);
          valLabel.setEnabled(true);
          valEntry.setEnabled(true);
        } else {
          presLabel.setEnabled(false);
          typeLabel.setEnabled(false);
          typeCB.setEnabled(false);
          matLabel.setEnabled(false);
          matCB.setEnabled(false);
          valLabel.setEnabled(false);
          valEntry.setEnabled(false);
        }
      }

      public void writeUintah(PrintWriter pw, String tab) {

        String tab1 = new String(tab+"  ");
        pw.println(tab+"<BCType id=\""+d_mat+"\" var=\""+d_type+"\" label=\""+
                     d_vector+ "\">");
        pw.println(tab1+"<value> ["+valEntry.x()+", "+valEntry.y()+", "+
                     valEntry.y()+"] </value>");
        pw.println(tab+"</BCType>");
      }

    }
  }
}
