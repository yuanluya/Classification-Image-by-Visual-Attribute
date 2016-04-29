We saved our result files and Python code to get the accuracy. You can also tune our neural net parameters and generate new prediction files and run same Python code to see the accuracy. 

1.Existing Result Files:

~/Classification-Image-by-Visual-Attribute/CNN_vgg/labels_name.json: labels of visual attribution, that is a 99 dimensional binary vector
~/Classification-Image-by-Visual-Attribute/CNN_vgg/labels_name_softmax.json: multilabel classification labels, 1-90
~/Classification-Image-by-Visual-Attribute/CNN_vgg/predications_name.json: predictions of visual attribution, 90 dimensional vectors
~/Classification-Image-by-Visual-Attribute/CNN_vgg/predications_name_softmax.json: multilabel classification predictions, 1-90
~/Classification-Image-by-Visual-Attribute/labels.json: ground truth label for each classes rather than for test images, this one should never be written or changed
~/Classification-Image-by-Visual-Attribute/test_names.txt: a list of image names of all the test images, should not be changed

2.Python code for accuracy checking:

in "~/Classification-Image-by-Visual-Attribute/" run 

python result_analysis.py ./CNN_vgg/predications_name.json ./CNN_vgg/labels_name.json

This will return the average distance between our prediction point and real point of all the test images in the vector space, the shape of the distance matrix of each image to all 90 classes, and the prediction accuracy (5 nearest neighbors). It will also generate a file with accuracy of each category

in "~/Classification-Image-by-Visual-Attribute/" run

python calculate_accuracy.py

This will return the accuracy of using VGG with softmax layer to do multilabel classification (5 nearest neighbors), and will generate a file with accuracy of each category.  

3. Caffe working code

You can replicate our result by running our Caffe code with our Caffe models. 

IMPORTANT: Make sure your caffe enabled Python layer to run the following code

in "~/Classification-Image-by-Visual-Attribute/CNN_vgg/"

run python cnn_vgg.py test

Our neural network will do prediction for you and write two files, labels_name.json and predications_name.json, which will be exactly the same with the existing files. 

run python cnn_vgg.py train

You can continue to train our neural network. But you need to go into the file and change the caffemodel loaded for test net so that it uses the latest model. If you want to starting from different caffemodel for vgg, change the model name in the 20 line. 

run python softmax.py

This code will do prediction with our trained softmax version of VGG. It will generate labels_name_softmax.json and predications_name_softmax.json. 

We used command line to fine tuning our softmax version VGG net. 
run 
"caffe -solver softmax_solver.prototxt -weights -VGG_ILSVRC_16_layers_softmax_30000.caffemodel"
to continue training our model starting with 30000 iterations. make sure your environment path know where is your caffe.bin. You can also run 
"caffe train -solver softmax_solver.prototxt -weights VGG_ILSVRC_16_layers.caffemodel"
to start training at the vanila VGG weights

4. Caffe Documents
in "~/Classification-Image-by-Visual-Attribute/CNN_vgg/"
There are six .prototxt files, they contain our neural network structures and solver file, their names should be self explaining. 
multilabel_layer.py and tools.py are helper functions for our neural networks, they should not be changed. 

5. Datasets
Our datasets live in ~/Classification-Image-by-Visual-Attribute/, they are seperated by train_images and test_images. 
The dataset we used as our visual attribute ground truth is in ANIMALS_structured_final.us.xml