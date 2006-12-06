//**************************************************************************
// Program : HelpAboutFrame.java
// Purpose : Create a general help screen
// Author  : Biswajit Banerjee
// Date    : 12/17/1998
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

//**************************************************************************
// Class   : HelpAboutFrame
// Purpose : About help text area
//**************************************************************************
public class HelpAboutFrame extends JFrame implements ActionListener {

    JTextArea textArea;

    public HelpAboutFrame() {

	// set the size
	setSize(300,300);
	setLocation(500,50);
	setTitle("About ...");

	// Create a gridbag and constraints
	GridBagLayout gb = new GridBagLayout();
	GridBagConstraints c = new GridBagConstraints();
	getContentPane().setLayout(gb);

	// Text Area
	UintahGui.setConstraints(c, GridBagConstraints.BOTH, 
					1.0, 1.0, 0, 0, 1, 1,1);
	String text = new String("Uintah User Interface");
	textArea = 
		new JTextArea(text,20,30);
	textArea.setEditable(false);
	textArea.setFont(new Font("Dialog",Font.PLAIN,12));
	createHelpText();
	gb.setConstraints(textArea, c);
	getContentPane().add(textArea);

	// Button
	UintahGui.setConstraints(c, GridBagConstraints.CENTER, 
					1.0, 1.0, 0, 1, 1, 1,1);
	Button button = new Button("Close Window");
	gb.setConstraints(button, c);
	button.addActionListener(this);
	getContentPane().add(button);
    }

    /** Respond to button pressed */
    public void actionPerformed(ActionEvent e) {
	setVisible(false);
    }

    // Create the help text for the About help
    private void createHelpText() {
	textArea.append("\n");
	textArea.append("Version 1.0\n");
	textArea.append("Author : Biswajit Banerjee\n");
	textArea.append("Department of Mechanical Engineering\n");
	textArea.append("University of Utah\n");
    }
}
