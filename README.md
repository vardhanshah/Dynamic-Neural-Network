Code is completely written in python3
python3 is required
tensorflow, numpy is required

to install python3: 		sudo apt install python3 python3-pip
to install tensorflow, numpy :  pip3 install tensorflow numpy

config file should be located in current directory

In config file,

all parameter inside config file should preserve below order,

	Parameters:
		Input_Filepath
		Input_Headers
		Input_Columns
		Output_Filepath
		Output_Headers
		Output_Columns
		Hidden_Layers
		Bias_Nodes
		Learning_Rate
		Epochs
		Mini_Batches
		Activation_Function 

No specifict name of the parameters required

By default, script will assume that given data, have regression type output
and calucaltes value of cost according to it.

If you have classification type output,
you need to append two parameters in config file

	Parameters:
		classification
		one_hot_encoding

set the value of classification parameter to be 'True' (without quotes)
if required output labels are not one_hot_encoded
 	set the value of one_hot_encoding parameter to be 'True'(without quotes), 
else 
	set the value of one_hot_encoding parameter to be 'False'(without quotes) or don't include last parameter

with this setting, code will conclude cost function according to what classification output required
and softmax function will be applied to output layer