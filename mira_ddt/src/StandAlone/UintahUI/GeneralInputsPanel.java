/**************************************************************************
// Program : GeneralInputsPanel.java
// Purpose : Create a panel that contains widgets to take inputs for
//           1) The title 
//           2) The simulation component
//           3) The time + timestep information
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************/

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;

//**************************************************************************
// Class   : GeneralInputsPanel
// Purpose : The following components are realized in the panel :
//             1) A gridbag layout that contains the components.
//             2) Text fields to take the inputs.
//             3) A button that saves the input data.
//**************************************************************************
public class GeneralInputsPanel extends JPanel {

  // Static variables

  // Data
  private String d_simComponent = null;
  private UintahInputPanel d_parent = null;

  // Two panels for time inputs and variable save inputs
  private PhysicalConstInputPanel constInputPanel = null;
  private TimeInputPanel timeInputPanel = null;
  private VariableSaveInputPanel saveInputPanel = null;

  private JButton saveButton = null;

  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------
  public GeneralInputsPanel(String simComponent,
                            UintahInputPanel parent) {

    // Initialize local variables
    d_simComponent = simComponent;
    d_parent = parent;

    // Create the panels
    timeInputPanel = new TimeInputPanel(d_simComponent, this);
    saveInputPanel = new VariableSaveInputPanel();
    constInputPanel = new PhysicalConstInputPanel();

    // Create the save button
    saveButton = new JButton("Save");
    saveButton.setActionCommand("save");

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Grid bag layout
    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(timeInputPanel, gbc);
    add(timeInputPanel);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(constInputPanel, gbc);
    add(constInputPanel);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 0, 2, 1, 1, 5);
    gb.setConstraints(saveButton, gbc);
    add(saveButton);

    UintahGui.setConstraints(gbc, GridBagConstraints.NONE, 
                             1.0, 1.0, 1, 0, 1, 3, 5);
    gb.setConstraints(saveInputPanel, gbc);
    add(saveInputPanel);

    // Add listener
    ButtonListener buttonListener = new ButtonListener();
    saveButton.addActionListener(buttonListener);
  }

  //-----------------------------------------------------------------------
  // Update the tabs in the parent panel
  //-----------------------------------------------------------------------
  public void updateTabs(String simComponent) {
    d_simComponent = simComponent;
    d_parent.enableTabs(simComponent);
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
    constInputPanel.refresh();
    timeInputPanel.refresh();
    saveInputPanel.refresh();
  }

  //-----------------------------------------------------------------------
  // Write out in Uintah format
  //-----------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {

    if (pw == null) return;

    timeInputPanel.writeUintah(pw, tab);
    saveInputPanel.writeUintah(pw, tab);
    constInputPanel.writeUintah(pw, tab);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Respond to button pressed (inner class button listener)
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class ButtonListener implements ActionListener {
    public void actionPerformed(ActionEvent e) {
      if (e.getActionCommand() == "save") {
        
        // Create filewriter and printwriter
        File outputFile = new File("test.ups");
        try {
          FileWriter fw = new FileWriter(outputFile);
          PrintWriter pw = new PrintWriter(fw);

          timeInputPanel.writeUintah(pw, "  ");
          saveInputPanel.writeUintah(pw, "  ");

          pw.close();
          fw.close();
        } catch (Exception event) {
          System.out.println("Could not write to file "+outputFile.getName());
        }
      }
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Physical constant inputs
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class PhysicalConstInputPanel extends JPanel {

    private DecimalField presEntry = null;
    private DecimalVectorField gravEntry = null;

    public PhysicalConstInputPanel() {

      // Create a grid bag for the components
      GridBagLayout gb = new GridBagLayout();
      GridBagConstraints gbc = new GridBagConstraints();
      setLayout(gb);
      int fill = GridBagConstraints.BOTH;
      int xgap = 5;
      int ygap = 0;

      // Reference pressure
      JLabel presLabel = new JLabel("Ref. Pressure");
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 0);
      gb.setConstraints(presLabel, gbc);
      add(presLabel);

      presEntry = new DecimalField(101325.0,9);
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 0);
      gb.setConstraints(presEntry, gbc);
      add(presEntry);

      // Gravity
      JLabel gravLabel = new JLabel("Gravity:");
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 0, 1);
      gb.setConstraints(gravLabel, gbc);
      add(gravLabel);

      gravEntry = new DecimalVectorField(0.0, 0.0, 0.0, 5);
      UintahGui.setConstraints(gbc, fill, xgap, ygap, 1, 1);
      gb.setConstraints(gravEntry, gbc);
      add(gravEntry);
    }

    public void refresh() {
    }

    public void writeUintah(PrintWriter pw, String tab) {

      if (pw == null) return;
     
      String tab1 = new String(tab+"  ");
      pw.println(tab+"<PhysicalConstants>");
      pw.println(tab1+"<reference_pressure> "+presEntry.getValue()+
		 " </reference_pressure>");
      pw.println(tab1+"<gravity> ["+gravEntry.x()+", "+
                 gravEntry.y()+", "+gravEntry.z()+
		 "] </gravity>");
      pw.println(tab+"</PhysicalConstants>");
      pw.println(tab);
    }  
  }

}
